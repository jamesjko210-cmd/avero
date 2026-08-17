from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from jarvis_v2.agent.planner import RuleBasedPlanner
from jarvis_v2.automations.compaction import COMPACTION_JOB_NAME, COMPACTION_JOB_TYPE
from jarvis_v2.scripts.test_runtime import make_temp_runtime
from jarvis_v2.memory.store import GoalRecord, MemoryRecord, PreferenceRecord, TaskRecord


READ_ONLY_FLAGS = [
    "calls_model",
    "executes_tools",
    "queues_approval",
    "requires_approval",
    "authorizes_execution",
    "authorizes_completion_claim",
    "approval_granted",
    "approves_request",
    "dismisses_request",
    "reads_personal_data",
    "reads_private_data",
    "executes_side_effect",
    "external_side_effect",
    "controls_computer",
    "writes_files",
    "writes_memory",
    "writes_notes",
    "speaks",
    "completes_tasks",
]
BACKGROUND_RHYTHM_KEYS = [
    "background_ready",
    "background_next_command",
    "background_issue",
    "background_priority",
    "state_snapshot_jobs",
    "enabled_state_snapshot_jobs",
    "disabled_state_snapshot_jobs",
    "conversation_compaction_jobs",
    "enabled_conversation_compaction_jobs",
    "disabled_conversation_compaction_jobs",
    "unreadable_scheduled_job_rows",
]


class HostileRow:
    def __init__(self, marker: str) -> None:
        self.marker = marker

    def __getitem__(self, key: str) -> object:
        raise RuntimeError(f"{self.marker}:{key}")

    def __str__(self) -> str:
        raise RuntimeError(self.marker)

    def __repr__(self) -> str:
        return f"<hostile-row {self.marker}>"


class HostileMetadataValue:
    def __init__(self, marker: str) -> None:
        self.marker = marker

    def __bool__(self) -> bool:
        raise RuntimeError(self.marker)

    def __str__(self) -> str:
        raise RuntimeError(self.marker)

    def __repr__(self) -> str:
        return f"<hostile-metadata {self.marker}>"


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
    if str(root) in receipt_line or "/private/" in receipt_line or "/\x55sers/" in receipt_line:
        raise SystemExit(f"{label} receipt should not expose local temp or user paths.")
    if not isinstance(path_display, str) or not path_display.startswith(prefix):
        raise SystemExit(f"{label} missed vault-relative display metadata: {metadata}")
    if path_display not in receipt_line:
        raise SystemExit(f"{label} should print the vault-relative saved-note label.")


def assert_brain_loop_handoff(metadata: dict, label: str) -> None:
    handoff = metadata.get("brain_loop_handoff")
    if not isinstance(handoff, dict):
        raise SystemExit(f"{label} missed brain_loop_handoff: {metadata}")
    if metadata.get("brain_loop_handoff_ready") is not True or handoff.get("handoff_ready") is not True:
        raise SystemExit(f"{label} should report handoff ready: {metadata}")
    if metadata.get("brain_loop_ready_for_operator") is not True or handoff.get("ready_for_operator") is not True:
        raise SystemExit(f"{label} should report ready_for_operator: {metadata}")
    if metadata.get("brain_loop_state_changed") is not False or handoff.get("state_changed") is not False:
        raise SystemExit(f"{label} should report unchanged state: {metadata}")
    if metadata.get("brain_loop_changed") != [] or handoff.get("changed") != []:
        raise SystemExit(f"{label} should report no changed resources: {metadata}")
    if metadata.get("brain_loop_content_in_handoff") is not False or handoff.get("content_in_handoff") is not False:
        raise SystemExit(f"{label} should keep detailed context out of the handoff: {metadata}")
    for key in [
        "profile_context",
        "recent_memories",
        "saved_skills",
        "active_goals",
        "open_tasks",
        "active_decisions",
        "active_preferences",
        "recent_tool_runs",
        "recent_ok_tool_runs",
        "recent_failed_tool_runs",
        "recent_approval_held_tool_runs",
        "pending_approvals",
        "enabled_jobs",
        "scheduled_jobs",
        "limit",
    ]:
        if handoff.get(key) != metadata.get(key):
            raise SystemExit(f"{label} handoff {key} diverged from metadata: {metadata}")
    for key in BACKGROUND_RHYTHM_KEYS:
        if handoff.get(key) != metadata.get(key):
            raise SystemExit(f"{label} background rhythm key {key} diverged from metadata: {metadata}")
    for flat_key, nested_key in [
        ("brain_loop_start_kind", "start_kind"),
        ("brain_loop_start_command", "start_command"),
        ("brain_loop_start_label", "start_label"),
        ("brain_loop_order", "loop_order"),
        ("brain_loop_next_commands", "next_commands"),
        ("brain_loop_next_safe_commands", "next_safe_commands"),
    ]:
        if metadata.get(flat_key) != handoff.get(nested_key):
            raise SystemExit(f"{label} {flat_key} diverged from nested handoff: {metadata}")
    if metadata.get("brain_loop_next_command_count") != len(handoff.get("next_commands") or []):
        raise SystemExit(f"{label} next-command count diverged: {metadata}")
    if metadata.get("brain_loop_next_safe_commands") != metadata.get("brain_loop_next_commands"):
        raise SystemExit(f"{label} safe next commands should mirror next commands: {metadata}")
    if handoff.get("next_safe_commands") != handoff.get("next_commands"):
        raise SystemExit(f"{label} nested safe next commands should mirror next commands: {metadata}")
    if metadata.get("brain_loop_next_safe_command_count") != len(handoff.get("next_safe_commands") or []):
        raise SystemExit(f"{label} safe next-command count diverged: {metadata}")
    for key in [
        "first_approval_id",
        "first_approval_tool",
        "first_readiness_command",
        "first_last_look_command",
        "first_proof_command",
        "first_approve_command",
        "first_dismiss_command",
    ]:
        if handoff.get(key) != metadata.get(key):
            raise SystemExit(f"{label} approval handoff field {key} diverged: {metadata}")
    if handoff.get("start_kind") not in {"approval_review", "task", "goal", "capture_task", "background_repair"}:
        raise SystemExit(f"{label} start kind is not a known resume category: {metadata}")
    for command in ["skill match preview: <request>", "assistant turn rehearsal: <message>", "rehearse: <request>", "focus brief", "work queue", "safe next actions"]:
        if command not in handoff.get("next_commands", []):
            raise SystemExit(f"{label} missed next command {command!r}: {metadata}")
    if metadata.get("pending_approvals") and not any(str(command).startswith("approval readiness") for command in handoff.get("next_commands") or []):
        raise SystemExit(f"{label} missed approval readiness next command: {metadata}")
    boundaries = handoff.get("boundaries") or {}
    if metadata.get("brain_loop_boundaries") != boundaries:
        raise SystemExit(f"{label} boundary packet diverged: {metadata}")
    if boundaries.get("read_only") is not True:
        raise SystemExit(f"{label} boundary should report read_only=True: {metadata}")
    for flag in READ_ONLY_FLAGS:
        if boundaries.get(flag) is not False:
            raise SystemExit(f"{label} boundary should report {flag}=False: {metadata}")
    for flat_key, boundary_key in [
        ("brain_loop_authorizes_execution", "authorizes_execution"),
        ("brain_loop_authorizes_completion_claim", "authorizes_completion_claim"),
        ("brain_loop_approval_granted", "approval_granted"),
    ]:
        if metadata.get(flat_key) is not False or metadata.get(flat_key) != boundaries.get(boundary_key):
            raise SystemExit(f"{label} {flat_key} should mirror false boundary flag: {metadata}")
    handoff_text = json.dumps(handoff, sort_keys=True)
    for forbidden in ["/\x55sers/", "/private/", "/var/folders/", "/tmp/"]:
        if forbidden in handoff_text:
            raise SystemExit(f"{label} handoff leaked local path fragment {forbidden!r}: {metadata}")


def assert_planner_routes_state_snapshot_aliases() -> None:
    planner = RuleBasedPlanner()
    for text in (
        "export state please",
        "state snapshot please",
        "current context snapshot please",
        "update current context please",
        # Real gaps found live 2026-07-09: "show state snapshot" fell through
        # to chat while bare "state snapshot" worked, and "save state" was
        # misrouted to the generic `remember` tool (treating "state" as a
        # literal memory to save) instead of `export_state_snapshot` -- same
        # collision class as the earlier "save decision" find.
        "show state snapshot",
        "save state",
    ):
        plan = planner.plan(text)
        if [(action.tool_name, action.args) for action in plan.actions] != [("export_state_snapshot", {})]:
            raise SystemExit(f"planner missed explicit state snapshot alias {text!r}: {plan.actions}")
    for text in ("show state", "state please", "current state please", "show current state", "show latest state snapshot"):
        plan = planner.plan(text)
        if plan.actions:
            raise SystemExit(f"planner should not route ambiguous state display alias {text!r}: {plan.actions}")


def assert_runtime_routes_control_count_aliases() -> None:
    with TemporaryDirectory(prefix="jarvis-control-count-aliases-") as temp:
        runtime = make_temp_runtime(Path(temp))
        cases = {
            "approval count": "list_pending_approvals",
            "count approvals": "list_pending_approvals",
            "how many approvals are pending": "list_pending_approvals",
            "scheduled job count": "list_scheduled_jobs",
            "count scheduled jobs": "list_scheduled_jobs",
            "how many scheduled jobs": "list_scheduled_jobs",
            "recent run count": "recent_tool_runs",
            "tool run count": "recent_tool_runs",
            "how many tool runs": "recent_tool_runs",
            "session count": "list_sessions",
            "count sessions": "list_sessions",
            "how many sessions": "list_sessions",
            "feedback count": "feedback_report",
            "count feedback": "feedback_report",
        }
        korean_cases = {
            "승인 몇 개": "list_pending_approvals",
            "승인 개수": "list_pending_approvals",
            "대기 승인 몇 개": "list_pending_approvals",
            "승인 대기 몇 개": "list_pending_approvals",
            "승인 큐 몇 개": "list_pending_approvals",
            "승인 필요한 거 몇 개": "list_pending_approvals",
            "예약 작업 몇 개": "list_scheduled_jobs",
            "스케줄 작업 몇 개": "list_scheduled_jobs",
            "예약된 작업 몇 개": "list_scheduled_jobs",
            "작업 예약 몇 개": "list_scheduled_jobs",
            "스케줄 몇 개": "list_scheduled_jobs",
            "최근 실행 몇 개": "recent_tool_runs",
            "실행 기록 몇 개": "recent_tool_runs",
            "도구 실행 몇 개": "recent_tool_runs",
            "최근 도구 실행 몇 개": "recent_tool_runs",
            "감사 로그 몇 개": "recent_tool_runs",
            "세션 몇 개": "list_sessions",
            "대화 몇 개": "list_sessions",
            "대화 세션 몇 개": "list_sessions",
            "피드백 몇 개": "feedback_report",
            "피드백 개수": "feedback_report",
        }
        cases.update(korean_cases)
        for text, expected_tool in cases.items():
            feedback_before = len(runtime.store.recent_memories(limit=1000))
            result = runtime.handle(text)
            tool_names = [tool_result.tool_name for tool_result in result.tool_results]
            if tool_names != [expected_tool]:
                raise SystemExit(f"runtime count alias {text!r} routed to {tool_names}, expected {[expected_tool]}")
            if not result.verified:
                raise SystemExit(f"runtime count alias {text!r} should be verified: {result.response}")
            metadata = result.tool_results[0].metadata
            for flag in ("queues_approval", "requires_approval", "authorizes_execution", "approval_granted", "writes_memory", "writes_notes", "writes_files", "writes_database", "external_side_effect", "controls_computer"):
                if metadata.get(flag) is True:
                    raise SystemExit(f"runtime count alias {text!r} should remain read-only, but {flag}=True: {metadata}")
            feedback_after = len(runtime.store.recent_memories(limit=1000))
            if expected_tool == "feedback_report" and feedback_after != feedback_before:
                raise SystemExit(f"feedback count alias should not record feedback: before={feedback_before}, after={feedback_after}")


def assert_state_tools_report_background_rhythm() -> None:
    with TemporaryDirectory(prefix="jarvis-state-bg-missing-compaction-") as temp:
        root = Path(temp)
        runtime = make_temp_runtime(root)
        runtime.store.upsert_job("State Snapshot", 360, "state_snapshot", "2099-01-01T00:00:00")

        brain = runtime.registry.get("brain_loop_report").handler({})
        if not brain.ok:
            raise SystemExit(f"brain_loop_report missing-compaction fixture failed: {brain.output}")
        if f"{COMPACTION_JOB_NAME} scheduled job is not configured." not in brain.output:
            raise SystemExit(f"brain_loop_report should flag missing Conversation Compaction: {brain.output}")
        if "schedule assistant basics" not in brain.output:
            raise SystemExit(f"brain_loop_report should recommend assistant basics for missing compaction: {brain.output}")
        if brain.metadata.get("background_ready") is not False:
            raise SystemExit(f"brain_loop_report should mark missing compaction background unready: {brain.metadata}")
        if brain.metadata.get("background_next_command") != "schedule assistant basics":
            raise SystemExit(f"brain_loop_report missed missing-compaction next command: {brain.metadata}")
        if brain.metadata.get("conversation_compaction_jobs") != 0 or brain.metadata.get("enabled_state_snapshot_jobs") != 1:
            raise SystemExit(f"brain_loop_report missed missing-compaction counters: {brain.metadata}")
        if brain.metadata.get("brain_loop_start_kind") != "background_repair":
            raise SystemExit(f"brain_loop_report should start with background repair when no work is ahead: {brain.metadata}")
        assert_brain_loop_handoff(brain.metadata, "missing compaction brain_loop_report")

        exported = runtime.registry.get("export_state_snapshot").handler({})
        assert_vault_relative_receipt(exported, root, "Memory Tree/", "missing compaction export_state_snapshot")
        exported_text = Path(exported.metadata["path"]).read_text(encoding="utf-8")
        for fragment in [
            "## Background Rhythm",
            f"{COMPACTION_JOB_NAME} scheduled job is not configured.",
            "Next safe command: `schedule assistant basics`.",
            "Conversation Compaction jobs: 0 total, 0 enabled",
        ]:
            if fragment not in exported_text:
                raise SystemExit(f"export_state_snapshot missed missing-compaction fragment {fragment!r}: {exported_text}")
        if exported.metadata.get("background_ready") is not False or exported.metadata.get("background_next_command") != "schedule assistant basics":
            raise SystemExit(f"export_state_snapshot missed missing-compaction metadata: {exported.metadata}")

    with TemporaryDirectory(prefix="jarvis-state-bg-paused-compaction-") as temp:
        root = Path(temp)
        runtime = make_temp_runtime(root)
        runtime.store.upsert_job("State Snapshot", 360, "state_snapshot", "2099-01-01T00:00:00")
        runtime.store.upsert_job(COMPACTION_JOB_NAME, 1440, COMPACTION_JOB_TYPE, "2099-01-01T00:00:00")
        if not runtime.store.set_job_enabled(COMPACTION_JOB_NAME, False):
            raise SystemExit("Paused compaction state fixture could not disable the compaction job.")

        brain = runtime.registry.get("brain_loop_report").handler({})
        expected_resume = f"resume job {COMPACTION_JOB_NAME}"
        if f"{COMPACTION_JOB_NAME} scheduled job is paused." not in brain.output or expected_resume not in brain.output:
            raise SystemExit(f"brain_loop_report should flag paused Conversation Compaction: {brain.output}")
        if (
            brain.metadata.get("background_ready") is not False
            or brain.metadata.get("background_next_command") != expected_resume
            or brain.metadata.get("conversation_compaction_jobs") != 1
            or brain.metadata.get("enabled_conversation_compaction_jobs") != 0
            or brain.metadata.get("disabled_conversation_compaction_jobs") != 1
        ):
            raise SystemExit(f"brain_loop_report missed paused-compaction metadata: {brain.metadata}")
        assert_brain_loop_handoff(brain.metadata, "paused compaction brain_loop_report")

        exported = runtime.registry.get("export_state_snapshot").handler({})
        assert_vault_relative_receipt(exported, root, "Memory Tree/", "paused compaction export_state_snapshot")
        exported_text = Path(exported.metadata["path"]).read_text(encoding="utf-8")
        for fragment in [
            f"{COMPACTION_JOB_NAME} scheduled job is paused.",
            f"Next safe command: `{expected_resume}`.",
            "Conversation Compaction jobs: 1 total, 0 enabled",
        ]:
            if fragment not in exported_text:
                raise SystemExit(f"export_state_snapshot missed paused-compaction fragment {fragment!r}: {exported_text}")
        if exported.metadata.get("background_next_command") != expected_resume or exported.metadata.get("disabled_conversation_compaction_jobs") != 1:
            raise SystemExit(f"export_state_snapshot missed paused-compaction metadata: {exported.metadata}")

    with TemporaryDirectory(prefix="jarvis-state-bg-healthy-compaction-") as temp:
        root = Path(temp)
        runtime = make_temp_runtime(root)
        runtime.store.upsert_job("State Snapshot", 360, "state_snapshot", "2099-01-01T00:00:00")
        runtime.store.upsert_job(COMPACTION_JOB_NAME, 1440, COMPACTION_JOB_TYPE, "2099-01-01T00:00:00")

        brain = runtime.registry.get("brain_loop_report").handler({})
        if f"State Snapshot and {COMPACTION_JOB_NAME} are enabled." not in brain.output:
            raise SystemExit(f"brain_loop_report should report healthy background rhythm: {brain.output}")
        if (
            brain.metadata.get("background_ready") is not True
            or brain.metadata.get("background_next_command") != "list scheduled jobs"
            or brain.metadata.get("enabled_state_snapshot_jobs") != 1
            or brain.metadata.get("enabled_conversation_compaction_jobs") != 1
            or brain.metadata.get("unreadable_scheduled_job_rows") != 0
        ):
            raise SystemExit(f"brain_loop_report missed healthy background metadata: {brain.metadata}")
        assert_brain_loop_handoff(brain.metadata, "healthy compaction brain_loop_report")

        exported = runtime.registry.get("export_state_snapshot").handler({})
        assert_vault_relative_receipt(exported, root, "Memory Tree/", "healthy compaction export_state_snapshot")
        exported_text = Path(exported.metadata["path"]).read_text(encoding="utf-8")
        if f"State Snapshot and {COMPACTION_JOB_NAME} are enabled." not in exported_text:
            raise SystemExit(f"export_state_snapshot should report healthy background rhythm: {exported_text}")
        if (
            exported.metadata.get("background_ready") is not True
            or exported.metadata.get("enabled_conversation_compaction_jobs") != 1
            or exported.metadata.get("unreadable_scheduled_job_rows") != 0
        ):
            raise SystemExit(f"export_state_snapshot missed healthy background metadata: {exported.metadata}")


def assert_state_tools_fail_closed_on_malformed_job_enabled_values() -> None:
    with TemporaryDirectory(prefix="jarvis-state-bg-malformed-enabled-") as temp:
        root = Path(temp)
        runtime = make_temp_runtime(root)
        original_list_jobs = runtime.store.list_jobs
        runtime.store.list_jobs = lambda: [
            {"name": "State Snapshot", "job_type": "state_snapshot", "enabled": "false", "next_run_at": "2099-01-01T00:00:00"},
            {"name": COMPACTION_JOB_NAME, "job_type": COMPACTION_JOB_TYPE, "enabled": "true", "next_run_at": "2099-01-02T00:00:00"},
        ]
        try:
            brain = runtime.registry.get("brain_loop_report").handler({})
            exported = runtime.registry.get("export_state_snapshot").handler({})
        finally:
            runtime.store.list_jobs = original_list_jobs

        expected_resume = "resume job State Snapshot"
        if "State Snapshot scheduled job is paused." not in brain.output or expected_resume not in brain.output:
            raise SystemExit(f"brain_loop_report should fail closed on malformed job enabled values: {brain.output}")
        for key, expected in {
            "background_ready": False,
            "background_next_command": expected_resume,
            "enabled_jobs": 0,
            "state_snapshot_jobs": 1,
            "enabled_state_snapshot_jobs": 0,
            "disabled_state_snapshot_jobs": 1,
            "conversation_compaction_jobs": 1,
            "enabled_conversation_compaction_jobs": 0,
            "disabled_conversation_compaction_jobs": 1,
            "unreadable_scheduled_job_rows": 0,
        }.items():
            if brain.metadata.get(key) != expected:
                raise SystemExit(f"brain_loop_report malformed-enabled {key} diverged: {brain.metadata}")
        assert_brain_loop_handoff(brain.metadata, "malformed enabled brain_loop_report")

        assert_vault_relative_receipt(exported, root, "Memory Tree/", "malformed enabled export_state_snapshot")
        exported_text = Path(exported.metadata["path"]).read_text(encoding="utf-8")
        for fragment in [
            "State Snapshot [state_snapshot, off] next 2099-01-01T00:00:00",
            f"{COMPACTION_JOB_NAME} [{COMPACTION_JOB_TYPE}, off] next 2099-01-02T00:00:00",
            "State Snapshot scheduled job is paused.",
            "State Snapshot jobs: 1 total, 0 enabled",
            "Conversation Compaction jobs: 1 total, 0 enabled",
        ]:
            if fragment not in exported_text:
                raise SystemExit(f"export_state_snapshot missed malformed-enabled fragment {fragment!r}: {exported_text}")
        for key, expected in {
            "background_ready": False,
            "background_next_command": expected_resume,
            "enabled_jobs": 0,
            "enabled_state_snapshot_jobs": 0,
            "disabled_state_snapshot_jobs": 1,
            "enabled_conversation_compaction_jobs": 0,
            "disabled_conversation_compaction_jobs": 1,
            "unreadable_scheduled_job_rows": 0,
        }.items():
            if exported.metadata.get(key) != expected:
                raise SystemExit(f"export_state_snapshot malformed-enabled {key} diverged: {exported.metadata}")


def assert_state_background_rhythm_fails_closed_on_unreadable_job_rows() -> None:
    marker = "STATE_BG_JOB_HOSTILE_SHOULD_NOT_LEAK /\x55sers/example/private/jobs.db"
    with TemporaryDirectory(prefix="jarvis-state-bg-unreadable-jobs-") as temp:
        root = Path(temp)
        runtime = make_temp_runtime(root)
        original_list_jobs = runtime.store.list_jobs
        runtime.store.list_jobs = lambda: [
            HostileRow(marker),
            {"name": "State Snapshot", "job_type": "state_snapshot", "enabled": True, "next_run_at": "2099-01-01T00:00:00"},
            {"name": COMPACTION_JOB_NAME, "job_type": COMPACTION_JOB_TYPE, "enabled": True, "next_run_at": "2099-01-02T00:00:00"},
        ]
        try:
            brain = runtime.registry.get("brain_loop_report").handler({})
            exported = runtime.registry.get("export_state_snapshot").handler({})
        finally:
            runtime.store.list_jobs = original_list_jobs

        if "Scheduled job state has unreadable rows." not in brain.output or "list scheduled jobs" not in brain.output:
            raise SystemExit(f"brain_loop_report should fail closed on unreadable scheduled jobs: {brain.output}")
        for key, expected in {
            "background_ready": False,
            "background_next_command": "list scheduled jobs",
            "background_issue": "Scheduled job state has unreadable rows.",
            "background_priority": "inspect scheduled jobs before trusting the background rhythm.",
            "unreadable_job_rows": 1,
            "unreadable_scheduled_job_rows": 1,
            "scheduled_jobs": 2,
            "enabled_jobs": 2,
            "enabled_state_snapshot_jobs": 1,
            "enabled_conversation_compaction_jobs": 1,
            "brain_loop_start_kind": "background_repair",
            "brain_loop_start_command": "list scheduled jobs",
        }.items():
            if brain.metadata.get(key) != expected:
                raise SystemExit(f"brain_loop_report unreadable-job {key} diverged: {brain.metadata}")
        assert_brain_loop_handoff(brain.metadata, "unreadable scheduled jobs brain_loop_report")

        assert_vault_relative_receipt(exported, root, "Memory Tree/", "unreadable scheduled jobs export_state_snapshot")
        exported_text = Path(exported.metadata["path"]).read_text(encoding="utf-8")
        for fragment in [
            "Hidden malformed local state rows: 1",
            "Scheduled job state has unreadable rows.",
            "Next safe command: `list scheduled jobs`.",
            "State Snapshot jobs: 1 total, 1 enabled",
            "Conversation Compaction jobs: 1 total, 1 enabled",
        ]:
            if fragment not in exported_text:
                raise SystemExit(f"export_state_snapshot missed unreadable-job fragment {fragment!r}: {exported_text}")
        combined = brain.output + exported_text + json.dumps(brain.metadata, sort_keys=True) + json.dumps(exported.metadata, sort_keys=True)
        for leaked in ["STATE_BG_JOB_HOSTILE_SHOULD_NOT_LEAK", "/\x55sers/example/private", "jobs.db"]:
            if leaked in combined:
                raise SystemExit(f"state background diagnostics leaked unreadable job marker {leaked!r}: {combined}")
        for key, expected in {
            "background_ready": False,
            "background_next_command": "list scheduled jobs",
            "background_issue": "Scheduled job state has unreadable rows.",
            "unreadable_job_rows": 1,
            "unreadable_scheduled_job_rows": 1,
            "scheduled_jobs": 2,
            "enabled_jobs": 2,
        }.items():
            if exported.metadata.get(key) != expected:
                raise SystemExit(f"export_state_snapshot unreadable-job {key} diverged: {exported.metadata}")


def assert_brain_loop_report_tolerates_malformed_local_rows() -> None:
    marker = "STATE_HOSTILE_ROW_SHOULD_NOT_LEAK /\x55sers/example/private/state.db"
    with TemporaryDirectory(prefix="jarvis-state-hostile-rows-") as temp:
        runtime = make_temp_runtime(Path(temp))
        original_recent_memories = runtime.store.recent_memories
        original_list_skills = runtime.store.list_active_skills
        original_list_goals = runtime.store.list_goals
        original_list_goal_steps = runtime.store.list_goal_steps
        original_list_tasks = runtime.store.list_tasks
        original_list_decisions = runtime.store.list_decisions
        original_list_preferences = runtime.store.list_preferences
        original_list_sessions = runtime.store.list_sessions
        original_recent_tool_runs = runtime.store.recent_tool_runs
        original_list_pending_approvals = runtime.store.list_pending_approvals
        original_list_jobs = runtime.store.list_jobs
        runtime.store.recent_memories = lambda limit=25: [
            HostileRow(marker),
            {"category": "facts", "title": "Readable memory"},
        ]
        runtime.store.list_active_skills = lambda limit=25: [
            HostileRow(marker),
            {"name": "Readable skill", "trigger": "state check", "body": "fallback", "review_status": "active"},
        ]
        runtime.store.list_goals = lambda status="active", limit=50: [
            HostileRow(marker),
            {"id": 17, "title": "Readable goal", "status": "active"},
        ]
        runtime.store.list_goal_steps = lambda goal_id: [
            HostileRow(marker),
            {"body": "Readable step", "status": "open"},
        ]
        runtime.store.list_tasks = lambda status="open", limit=50: [
            HostileRow(marker),
            {"id": 23, "body": "Readable task"},
        ]
        runtime.store.list_decisions = lambda status="active", limit=3: [
            HostileRow(marker),
            {"id": 31, "title": "Readable decision"},
        ]
        runtime.store.list_preferences = lambda status="active", limit=5: [
            HostileRow(marker),
            {"key": "tone", "value": "concise"},
        ]
        runtime.store.list_sessions = lambda limit=20: [
            HostileRow(marker),
            {"session_id": "session-readable", "messages": 3, "last_at": "2099-01-01T00:00:00"},
        ]
        runtime.store.recent_tool_runs = lambda limit=25: [
            HostileRow(marker),
            {"tool_name": "readiness_report", "risk": "LOCAL_SAFE", "ok": True},
        ]
        runtime.store.list_pending_approvals = lambda status="pending", limit=25: [
            HostileRow(marker),
            {"id": 41, "tool_name": "send_telegram"},
        ]
        runtime.store.list_jobs = lambda: [
            HostileRow(marker),
            {"name": "State Snapshot", "job_type": "state_snapshot", "enabled": True},
            {"name": COMPACTION_JOB_NAME, "job_type": COMPACTION_JOB_TYPE, "enabled": True},
        ]
        try:
            report = runtime.registry.get("brain_loop_report").handler({})
        finally:
            runtime.store.recent_memories = original_recent_memories
            runtime.store.list_active_skills = original_list_skills
            runtime.store.list_goals = original_list_goals
            runtime.store.list_goal_steps = original_list_goal_steps
            runtime.store.list_tasks = original_list_tasks
            runtime.store.list_decisions = original_list_decisions
            runtime.store.list_preferences = original_list_preferences
            runtime.store.list_sessions = original_list_sessions
            runtime.store.recent_tool_runs = original_recent_tool_runs
            runtime.store.list_pending_approvals = original_list_pending_approvals
            runtime.store.list_jobs = original_list_jobs

        if not report.ok:
            raise SystemExit(f"brain_loop_report should tolerate malformed local rows: {report.output}")
        for expected in [
            "hidden malformed local state rows: 11",
            "Readable memory",
            "Readable skill",
            "Readable task",
            "Readable goal",
            "Readable step",
            "Readable decision",
            "tone = concise",
            "approval readiness 41",
            "Scheduled job state has unreadable rows.",
            "list scheduled jobs",
        ]:
            if expected not in report.output:
                raise SystemExit(f"malformed-row brain_loop_report missed {expected!r}: {report.output}")
        leak_text = report.output + json.dumps(report.metadata, sort_keys=True)
        for leaked in ["STATE_HOSTILE_ROW_SHOULD_NOT_LEAK", "/\x55sers/example/private", "state.db"]:
            if leaked in leak_text:
                raise SystemExit(f"brain_loop_report leaked hostile row marker {leaked!r}: {leak_text}")
        if report.metadata.get("unreadable_state_rows") != 11:
            raise SystemExit(f"brain_loop_report missed unreadable row count: {report.metadata}")
        for key in [
            "unreadable_memory_rows",
            "unreadable_skill_rows",
            "unreadable_goal_rows",
            "unreadable_goal_step_rows",
            "unreadable_task_rows",
            "unreadable_decision_rows",
            "unreadable_preference_rows",
            "unreadable_session_rows",
            "unreadable_tool_run_rows",
            "unreadable_approval_rows",
            "unreadable_job_rows",
        ]:
            if report.metadata.get(key) != 1:
                raise SystemExit(f"brain_loop_report missed {key}: {report.metadata}")
        expected_counts = {
            "recent_memories": 1,
            "saved_skills": 1,
            "active_goals": 1,
            "open_tasks": 1,
            "active_decisions": 1,
            "active_preferences": 1,
            "recent_sessions": 1,
            "recent_tool_runs": 1,
            "pending_approvals": 1,
            "scheduled_jobs": 2,
            "enabled_jobs": 2,
        }
        for key, expected in expected_counts.items():
            if report.metadata.get(key) != expected:
                raise SystemExit(f"brain_loop_report readable count {key} diverged: {report.metadata}")
        if report.metadata.get("first_readiness_command") != "approval readiness 41":
            raise SystemExit(f"brain_loop_report missed readable approval handoff: {report.metadata}")
        for key, expected in {
            "background_ready": False,
            "background_next_command": "list scheduled jobs",
            "background_issue": "Scheduled job state has unreadable rows.",
            "unreadable_scheduled_job_rows": 1,
        }.items():
            if report.metadata.get(key) != expected:
                raise SystemExit(f"brain_loop_report malformed-row background {key} diverged: {report.metadata}")
        assert_brain_loop_handoff(report.metadata, "malformed-row brain_loop_report")


def assert_export_state_snapshot_tolerates_malformed_local_rows() -> None:
    marker = "STATE_EXPORT_HOSTILE_ROW_SHOULD_NOT_LEAK /\x55sers/example/private/export-state.db"
    with TemporaryDirectory(prefix="jarvis-state-export-hostile-") as temp:
        root = Path(temp)
        runtime = make_temp_runtime(root)
        original_recent_memories = runtime.store.recent_memories
        original_list_goals = runtime.store.list_goals
        original_list_goal_steps = runtime.store.list_goal_steps
        original_list_tasks = runtime.store.list_tasks
        original_list_decisions = runtime.store.list_decisions
        original_list_people = runtime.store.list_people
        original_list_preferences = runtime.store.list_preferences
        original_list_sessions = runtime.store.list_sessions
        original_recent_tool_runs = runtime.store.recent_tool_runs
        original_list_pending_approvals = runtime.store.list_pending_approvals
        original_list_jobs = runtime.store.list_jobs
        runtime.store.recent_memories = lambda limit=25: [
            HostileRow(marker),
            {"category": "facts", "title": "Readable export memory"},
        ]
        runtime.store.list_goals = lambda status=None, limit=50: [
            HostileRow(marker),
            {"id": 17, "title": "Readable export goal", "status": "active"},
        ]
        runtime.store.list_goal_steps = lambda goal_id: [
            HostileRow(marker),
            {"body": "Readable export step", "status": "open"},
        ]
        runtime.store.list_tasks = lambda status="open", limit=50: [
            HostileRow(marker),
            {"id": 23, "body": "Readable export task", "due": "2099-01-01"},
        ]
        runtime.store.list_decisions = lambda status="active", limit=8: [
            HostileRow(marker),
            {"id": 31, "title": "Readable export decision", "rationale": "keep exports safe"},
        ]
        runtime.store.list_people = lambda limit=8: [
            HostileRow(marker),
            {"id": 37, "name": "Readable Person", "relation": "test contact", "last_contact_at": "2099-01-02"},
        ]
        runtime.store.list_preferences = lambda status="active", limit=12: [
            HostileRow(marker),
            {"category": "communication", "key": "tone", "value": "concise"},
        ]
        runtime.store.list_sessions = lambda limit=8: [
            HostileRow(marker),
            {"session_id": "session-readable", "messages": 3, "last_at": "2099-01-03T00:00:00"},
        ]
        runtime.store.recent_tool_runs = lambda limit=8: [
            HostileRow(marker),
            {"tool_name": "readiness_report", "risk": "READ_ONLY", "ok": True, "created_at": "2099-01-04T00:00:00"},
        ]
        runtime.store.list_pending_approvals = lambda status="pending", limit=8: [
            HostileRow(marker),
            {"id": 51, "tool_name": "send_telegram", "user_input": "send readable test"},
        ]
        runtime.store.list_jobs = lambda: [
            HostileRow(marker),
            {"name": "State Snapshot", "job_type": "state_snapshot", "enabled": True, "next_run_at": "2099-01-05T00:00:00"},
            {"name": COMPACTION_JOB_NAME, "job_type": COMPACTION_JOB_TYPE, "enabled": True, "next_run_at": "2099-01-06T00:00:00"},
        ]
        try:
            exported = runtime.registry.get("export_state_snapshot").handler({})
        finally:
            runtime.store.recent_memories = original_recent_memories
            runtime.store.list_goals = original_list_goals
            runtime.store.list_goal_steps = original_list_goal_steps
            runtime.store.list_tasks = original_list_tasks
            runtime.store.list_decisions = original_list_decisions
            runtime.store.list_people = original_list_people
            runtime.store.list_preferences = original_list_preferences
            runtime.store.list_sessions = original_list_sessions
            runtime.store.recent_tool_runs = original_recent_tool_runs
            runtime.store.list_pending_approvals = original_list_pending_approvals
            runtime.store.list_jobs = original_list_jobs

        if not exported.ok:
            raise SystemExit(f"export_state_snapshot should tolerate malformed local rows: {exported.output}")
        assert_vault_relative_receipt(exported, root, "Memory Tree/", "malformed-row export_state_snapshot")
        exported_text = Path(exported.metadata["path"]).read_text(encoding="utf-8")
        combined = exported_text + json.dumps(exported.metadata, sort_keys=True)
        for leaked in ["STATE_EXPORT_HOSTILE_ROW_SHOULD_NOT_LEAK", "/\x55sers/example/private", "export-state.db"]:
            if leaked in combined:
                raise SystemExit(f"export_state_snapshot leaked hostile row marker {leaked!r}: {combined}")
        for expected in [
            "Hidden malformed local state rows: 11",
            "Readable export memory",
            "Readable export goal: Readable export step",
            "Readable export task due 2099-01-01",
            "Readable export decision: keep exports safe",
            "Readable Person (test contact), last contact 2099-01-02",
            "[communication] tone: concise",
            "readiness_report [READ_ONLY, ok] 2099-01-04T00:00:00",
            "#51 send_telegram: send readable test",
            "approval chain proof 51",
            "Scheduled job state has unreadable rows.",
            "Next safe command: `list scheduled jobs`.",
        ]:
            if expected not in exported_text:
                raise SystemExit(f"malformed-row export_state_snapshot missed {expected!r}: {exported_text}")
        metadata = exported.metadata
        if metadata.get("unreadable_state_rows") != 11 or metadata.get("readable_state_rows") != 12:
            raise SystemExit(f"export_state_snapshot state row counts diverged: {metadata}")
        for key in [
            "unreadable_memory_rows",
            "unreadable_goal_rows",
            "unreadable_goal_step_rows",
            "unreadable_task_rows",
            "unreadable_decision_rows",
            "unreadable_people_rows",
            "unreadable_preference_rows",
            "unreadable_session_rows",
            "unreadable_tool_run_rows",
            "unreadable_approval_rows",
            "unreadable_job_rows",
        ]:
            if metadata.get(key) != 1:
                raise SystemExit(f"export_state_snapshot missed {key}: {metadata}")
        if metadata.get("first_readiness_command") != "approval readiness 51":
            raise SystemExit(f"export_state_snapshot missed readable approval handoff: {metadata}")
        if metadata.get("pending_approvals") != 1 or metadata.get("scheduled_jobs") != 2 or metadata.get("enabled_jobs") != 2:
            raise SystemExit(f"export_state_snapshot readable counts diverged: {metadata}")
        for key, expected in {
            "background_ready": False,
            "background_next_command": "list scheduled jobs",
            "background_issue": "Scheduled job state has unreadable rows.",
            "unreadable_scheduled_job_rows": 1,
        }.items():
            if metadata.get(key) != expected:
                raise SystemExit(f"export_state_snapshot malformed-row background {key} diverged: {metadata}")
        for key in READ_ONLY_FLAGS:
            if key in {"writes_files", "writes_notes", "reads_personal_data", "reads_private_data"}:
                if metadata.get(key) is not True:
                    raise SystemExit(f"export_state_snapshot should declare truthful {key}: {metadata}")
            elif metadata.get(key) is not False:
                raise SystemExit(f"export_state_snapshot unsafe metadata {key}: {metadata}")


def assert_state_tools_separate_approval_held_tool_runs() -> None:
    with TemporaryDirectory(prefix="jarvis-state-approval-held-") as temp:
        root = Path(temp)
        runtime = make_temp_runtime(root)
        runtime.store.log_tool_run(
            "state-approval-held",
            "list_tools",
            "READ_ONLY",
            True,
            True,
            "listed tools",
            metadata={},
        )
        runtime.store.log_tool_run(
            "state-approval-held",
            "send_kakao",
            "HIGH_RISK",
            False,
            True,
            "transport timed out",
            metadata={"failure_stage": "transport_timeout"},
        )
        runtime.store.log_tool_run(
            "state-approval-held",
            "send_telegram",
            "HIGH_RISK",
            False,
            False,
            "blocked before execution; queued as approval #4",
            approval_id=4,
            metadata={"failure_kind": "approval-gate"},
        )

        brain = runtime.registry.get("brain_loop_report").handler({"limit": 6})
        if not brain.ok:
            raise SystemExit(f"brain_loop_report approval-held fixture failed: {brain.output}")
        expected_summary = "recent tool runs: 3 (1 ok, 1 failed/blocked, 1 approval-held)"
        if expected_summary not in brain.output:
            raise SystemExit(f"brain_loop_report should separate approval-held runs: {brain.output}")
        if "latest run: send_telegram [HIGH_RISK, approval held]" not in brain.output:
            raise SystemExit(f"brain_loop_report should label latest approval-held run: {brain.output}")
        if "approval-gate" in brain.output:
            raise SystemExit(f"brain_loop_report should not expose raw approval metadata: {brain.output}")
        for key, expected in {
            "recent_tool_runs": 3,
            "recent_ok_tool_runs": 1,
            "recent_failed_tool_runs": 1,
            "recent_approval_held_tool_runs": 1,
        }.items():
            if brain.metadata.get(key) != expected:
                raise SystemExit(f"brain_loop_report missed {key}={expected}: {brain.metadata}")
        assert_brain_loop_handoff(brain.metadata, "approval-held brain_loop_report")

        exported = runtime.registry.get("export_state_snapshot").handler({})
        assert_vault_relative_receipt(exported, root, "Memory Tree/", "approval-held export_state_snapshot")
        exported_text = Path(exported.metadata["path"]).read_text(encoding="utf-8")
        for expected in [
            "send_telegram [HIGH_RISK, approval held]",
            "send_kakao [HIGH_RISK, failed]",
            "list_tools [READ_ONLY, ok]",
        ]:
            if expected not in exported_text:
                raise SystemExit(f"export_state_snapshot missed tool-run status {expected!r}: {exported_text}")
        if "approval-gate" in exported_text:
            raise SystemExit(f"export_state_snapshot should not expose raw approval metadata: {exported_text}")
        for key, expected in {
            "recent_tool_runs": 3,
            "recent_ok_tool_runs": 1,
            "recent_failed_tool_runs": 1,
            "recent_approval_held_tool_runs": 1,
        }.items():
            if exported.metadata.get(key) != expected:
                raise SystemExit(f"export_state_snapshot missed {key}={expected}: {exported.metadata}")


def assert_state_tools_classify_structured_tool_run_metadata() -> None:
    marker = "STATE_STRUCTURED_METADATA_SHOULD_NOT_LEAK"
    with TemporaryDirectory(prefix="jarvis-state-structured-metadata-") as temp:
        root = Path(temp)
        runtime = make_temp_runtime(root)
        original_recent_tool_runs = runtime.store.recent_tool_runs
        runtime.store.recent_tool_runs = lambda limit=8: [
            {
                "tool_name": "send_telegram",
                "risk": "HIGH_RISK",
                "ok": False,
                "created_at": "2099-01-01T00:00:00",
                "metadata": {"failure_kind": "approval-gate"},
            },
            {
                "tool_name": "send_kakao",
                "risk": "HIGH_RISK",
                "ok": False,
                "created_at": "2099-01-01T00:00:01",
                "metadata": {"failure_stage": HostileMetadataValue(marker)},
            },
            {
                "tool_name": "list_tools",
                "risk": "READ_ONLY",
                "ok": True,
                "created_at": "2099-01-01T00:00:02",
                "metadata": {},
            },
        ]
        try:
            brain = runtime.registry.get("brain_loop_report").handler({"limit": 8})
            exported = runtime.registry.get("export_state_snapshot").handler({})
        finally:
            runtime.store.recent_tool_runs = original_recent_tool_runs

        if not brain.ok:
            raise SystemExit(f"brain_loop_report structured metadata fixture failed: {brain.output}")
        expected_summary = "recent tool runs: 3 (1 ok, 1 failed/blocked, 1 approval-held)"
        if expected_summary not in brain.output:
            raise SystemExit(f"brain_loop_report missed structured metadata approval-held split: {brain.output}")
        if "latest run: send_telegram [HIGH_RISK, approval held]" not in brain.output:
            raise SystemExit(f"brain_loop_report missed structured approval-held label: {brain.output}")
        for key, expected in {
            "recent_tool_runs": 3,
            "recent_ok_tool_runs": 1,
            "recent_failed_tool_runs": 1,
            "recent_approval_held_tool_runs": 1,
        }.items():
            if brain.metadata.get(key) != expected:
                raise SystemExit(f"brain_loop_report structured metadata missed {key}={expected}: {brain.metadata}")

        if not exported.ok:
            raise SystemExit(f"export_state_snapshot structured metadata fixture failed: {exported.output}")
        assert_vault_relative_receipt(exported, root, "Memory Tree/", "structured metadata export_state_snapshot")
        exported_text = Path(exported.metadata["path"]).read_text(encoding="utf-8")
        for expected in [
            "send_telegram [HIGH_RISK, approval held]",
            "send_kakao [HIGH_RISK, failed]",
            "list_tools [READ_ONLY, ok]",
        ]:
            if expected not in exported_text:
                raise SystemExit(f"export_state_snapshot missed structured metadata status {expected!r}: {exported_text}")
        combined = brain.output + exported_text + json.dumps(brain.metadata, sort_keys=True) + json.dumps(exported.metadata, sort_keys=True)
        if marker in combined or "approval-gate" in combined:
            raise SystemExit(f"state diagnostics leaked raw structured metadata: {combined}")
        for key, expected in {
            "recent_tool_runs": 3,
            "recent_ok_tool_runs": 1,
            "recent_failed_tool_runs": 1,
            "recent_approval_held_tool_runs": 1,
        }.items():
            if exported.metadata.get(key) != expected:
                raise SystemExit(f"export_state_snapshot structured metadata missed {key}={expected}: {exported.metadata}")


def main() -> None:
    assert_planner_routes_state_snapshot_aliases()
    assert_runtime_routes_control_count_aliases()
    assert_state_tools_report_background_rhythm()
    assert_state_tools_fail_closed_on_malformed_job_enabled_values()
    assert_state_background_rhythm_fails_closed_on_unreadable_job_rows()
    assert_brain_loop_report_tolerates_malformed_local_rows()
    assert_export_state_snapshot_tolerates_malformed_local_rows()
    assert_state_tools_separate_approval_held_tool_runs()
    assert_state_tools_classify_structured_tool_run_metadata()

    with TemporaryDirectory(prefix="jarvis-state-") as temp:
        root = Path(temp)
        runtime = make_temp_runtime(root)
        cases = [
            "remember that state snapshots keep Jarvis oriented",
            "create goal Keep Jarvis oriented because persist current context",
            "add step to goal 1: export current context",
            "schedule assistant basics",
            "calculate 3 + 4",
            "run command python3 --version",
            "brain loop",
            "export state",
        ]
        for case in cases:
            result = runtime.handle(case)
            status = "ok" if result.verified else "blocked"
            print(f"[{status}] {case}")
            print(result.response[:1200])
            print()
            if case == "brain loop":
                required = [
                    "Jarvis brain loop report",
                    "Perception and conversation",
                    "Memory and skills",
                    "Planning and current work",
                    "Execution boundary",
                    "Verification and audit",
                    "Good next checks",
                    "skill match preview",
                    "assistant turn rehearsal",
                    "read-only",
                    "first safe approval handoff",
                    "approval readiness 1",
                    "approval packet 1",
                    "approval chain proof 1",
                ]
                missing = [item for item in required if item not in result.response]
                if missing:
                    raise SystemExit(f"Brain loop report missing expected text: {missing}")
                metadata = result.tool_results[0].metadata
                expected_handoff = {
                    "first_approval_id": 1,
                    "first_approval_tool": "run_shell_command",
                    "first_readiness_command": "approval readiness 1",
                    "first_last_look_command": "approval packet 1",
                    "first_proof_command": "approval chain proof 1",
                    "first_approve_command": "approve approval 1",
                    "first_dismiss_command": "dismiss approval 1",
                }
                for key, expected in expected_handoff.items():
                    if metadata.get(key) != expected:
                        raise SystemExit(f"Brain loop report missed approval handoff {key}: {metadata}")
                assert_brain_loop_handoff(metadata, "brain loop")
                if (
                    metadata.get("calls_model")
                    or metadata.get("executes_tools")
                    or metadata.get("authorizes_execution")
                    or metadata.get("authorizes_completion_claim")
                    or metadata.get("approval_granted")
                    or metadata.get("writes_memory")
                    or metadata.get("controls_computer")
                    or metadata.get("queues_approval")
                    or metadata.get("writes_files")
                ):
                    raise SystemExit("Brain loop report must not execute tools, write memory/files, control the computer, call models, or queue approvals.")
            if case == "export state":
                assert_vault_relative_receipt(result, root, "Memory Tree/", case)
                metadata = result.tool_results[0].metadata
                if not metadata.get("writes_files") or not metadata.get("writes_notes") or metadata.get("writes_memory"):
                    raise SystemExit(f"State snapshot should declare only its local note write: {metadata}")
                if metadata.get("reads_personal_data") is not True or metadata.get("reads_private_data") is not True:
                    raise SystemExit(f"State snapshot should declare its private local-state reads: {metadata}")
                if metadata.get("write_atomic") is not True or metadata.get("snapshot_consistency") != "transaction_fenced":
                    raise SystemExit(f"State snapshot missed durable consistency evidence: {metadata}")
                if metadata.get("first_readiness_command") != "approval readiness 1" or metadata.get("first_proof_command") != "approval chain proof 1":
                    raise SystemExit(f"State snapshot missed approval handoff metadata: {metadata}")
                for key in [
                    "calls_model",
                    "executes_tools",
                    "queues_approval",
                    "authorizes_execution",
                    "authorizes_completion_claim",
                    "approval_granted",
                    "controls_computer",
                    "executes_side_effect",
                ]:
                    if metadata.get(key):
                        raise SystemExit(f"State snapshot unsafe metadata {key}: {metadata}")

        path = root / "Vault" / "Jarvis" / "Memory Tree" / "Current Context.md"
        text = path.read_text(encoding="utf-8")
        if "Keep Jarvis oriented" not in text or "Recent Tool Runs" not in text or "First safe approval handoff" not in text or "approval chain proof 1" not in text:
            raise SystemExit("State snapshot did not include expected content.")

        bounded = runtime.registry.get("brain_loop_report").handler({"limit": 999})
        print("[ok] direct bounded brain_loop_report")
        print(bounded.output[:900])
        print()
        if not bounded.ok or bounded.metadata.get("limit") != 25:
            raise SystemExit(f"Brain loop limit was not bounded: {bounded.metadata}")
        assert_brain_loop_handoff(bounded.metadata, "bounded brain_loop_report")

        bad_limit = runtime.registry.get("brain_loop_report").handler({"limit": "bad"})
        if not bad_limit.ok or bad_limit.metadata.get("limit") != 6 or bad_limit.metadata.get("raw_limit") != "bad":
            raise SystemExit(f"Brain loop bad limit did not preserve sanitized raw metadata: {bad_limit.metadata}")
        assert_brain_loop_handoff(bad_limit.metadata, "bad limit brain_loop_report")
        bool_limit = runtime.registry.get("brain_loop_report").handler({"limit": False})
        if not bool_limit.ok or bool_limit.metadata.get("limit") != 6 or bool_limit.metadata.get("raw_limit") != "False":
            raise SystemExit(f"Brain loop should treat boolean limits as malformed and preserve raw metadata: {bool_limit.metadata}")
        assert_brain_loop_handoff(bool_limit.metadata, "boolean limit brain_loop_report")
        long_bad_limit = runtime.registry.get("brain_loop_report").handler({"limit": "l" * 120})
        if long_bad_limit.metadata.get("raw_limit") != ("l" * 79 + "…"):
            raise SystemExit(f"Brain loop long bad limit was not bounded: {long_bad_limit.metadata}")
        path_bad_limit = runtime.registry.get("brain_loop_report").handler({"limit": "/private/tmp/jarvis-state-limit"})
        if not path_bad_limit.ok or path_bad_limit.metadata.get("limit") != 6 or path_bad_limit.metadata.get("raw_limit") != "<local-path>":
            raise SystemExit(f"Brain loop should redact path-shaped bad limits: {path_bad_limit.metadata}")
        var_bad_limit = runtime.registry.get("brain_loop_report").handler({"limit": "/var/folders/zc/jarvis-state-limit"})
        if not var_bad_limit.ok or var_bad_limit.metadata.get("limit") != 6 or var_bad_limit.metadata.get("raw_limit") != "<local-path>":
            raise SystemExit(f"Brain loop should redact macOS temp-root bad limits: {var_bad_limit.metadata}")
        tmp_bad_limit = runtime.registry.get("brain_loop_report").handler({"limit": "/tmp/jarvis-state-limit"})
        if not tmp_bad_limit.ok or tmp_bad_limit.metadata.get("limit") != 6 or tmp_bad_limit.metadata.get("raw_limit") != "<local-path>":
            raise SystemExit(f"Brain loop should redact tmp-root bad limits: {tmp_bad_limit.metadata}")

        runtime.store.add_memory(MemoryRecord("facts", "/\x55sers/example/Desktop/Claude code/state-memory.md", "state snapshot path smoke"))
        runtime.store.add_memory(MemoryRecord("facts", "/var/folders/zc/state-memory.md", "state snapshot var path smoke"))
        runtime.store.add_task(TaskRecord("/private/tmp/state-task.md should not leak"))
        runtime.store.add_task(TaskRecord("/tmp/state-task.md should not leak"))
        runtime.store.create_goal(GoalRecord("/\x55sers/example/Desktop/state-goal.md", "state path smoke"))
        runtime.store.set_preference(PreferenceRecord("/\x55sers/example/Desktop/state-key", "/private/tmp/state-value"))

        path_report = runtime.registry.get("brain_loop_report").handler({"limit": 25})
        if any(fragment in path_report.output for fragment in ["/\x55sers/", "/private/", "/var/folders/", "/tmp/"]) or "<local-path>" not in path_report.output:
            raise SystemExit(f"Brain loop report should redact path-shaped stored records: {path_report.output}")
        assert_brain_loop_handoff(path_report.metadata, "path-redacted brain_loop_report")

        path_export = runtime.registry.get("export_state_snapshot").handler({})
        assert_vault_relative_receipt(path_export, root, "Memory Tree/", "export_state_snapshot direct")
        path_text = Path(path_export.metadata["path"]).read_text(encoding="utf-8")
        if any(fragment in path_text for fragment in ["/\x55sers/", "/private/", "/var/folders/", "/tmp/"]) or "<local-path>" not in path_text:
            raise SystemExit("State snapshot export should redact path-shaped stored records.")


if __name__ == "__main__":
    main()
