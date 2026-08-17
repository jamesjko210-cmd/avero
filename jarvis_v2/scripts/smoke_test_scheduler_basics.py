from __future__ import annotations

import hashlib
import json
import time
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path
from queue import Queue
from tempfile import TemporaryDirectory
from threading import Barrier, Event, Lock, Thread, current_thread
from unittest import mock

from jarvis_v2.agent.planner import RuleBasedPlanner
from jarvis_v2.agent.types import Plan, PlannedAction
from jarvis_v2.agent.verifier import Verifier
from jarvis_v2.automations.scheduler import Scheduler, _run_marker, iso
from jarvis_v2.memory.store import (
    MemoryRecord,
    ScheduledJobLeaseAuthorityLost,
)
from jarvis_v2.scripts.test_runtime import handle_runtime_case, make_temp_runtime
from jarvis_v2.tools.registry import TOOL_ARGUMENT_CONTRACT_VERSION
from jarvis_v2.tools.scheduler import MAX_JOB_INTERVAL_MINUTES, MAX_JOB_NAME_CHARS


SAFE_FALSE_FLAGS = [
    "calls_model",
    "executes_tools",
    "reads_private_data",
    "reads_personal_data",
    "writes_memory",
    "external_side_effect",
    "queues_approval",
    "requires_approval",
    "authorizes_execution",
    "authorizes_completion_claim",
    "approval_granted",
    "controls_computer",
    "speaks",
    "completes_tasks",
]


def assert_scheduler_recovery_guidance(output: str, *, delivery_receipt: bool = False) -> None:
    for expected in (
        "list scheduled jobs",
        "setup check",
        "storage, model, or connector issue",
    ):
        if expected not in output:
            raise SystemExit(f"scheduler failure missed recovery guidance {expected!r}: {output}")
    if delivery_receipt:
        for expected in (
            "durable delivery receipt state",
            "do not manually rerun or resend",
            "outcome-unknown delivery",
        ):
            if expected not in output:
                raise SystemExit(
                    f"scheduler delivery failure missed safe receipt guidance {expected!r}: {output}"
                )


class HostileJobRow:
    def __init__(self, marker: str):
        self.marker = marker

    def __getitem__(self, key: str) -> object:
        raise RuntimeError(f"{self.marker}:{key}")

    def __str__(self) -> str:
        return self.marker


class HostileTruthinessText:
    def __init__(self, text: str) -> None:
        self.text = text

    def __bool__(self) -> bool:
        raise RuntimeError("hostile truthiness should not be evaluated")

    def __str__(self) -> str:
        return self.text


class _StaticPlanner:
    def __init__(self, args: object) -> None:
        self.args = args

    def plan(self, _user_input: str) -> Plan:
        return Plan(
            "Exercise the scheduled-job-list runtime argument contract.",
            [PlannedAction("list_scheduled_jobs", self.args, "scheduled-job-list contract smoke")],
            needs_model=False,
        )


def assert_no_local_path(value: object, label: str) -> None:
    text = str(value)
    for fragment in ("/\x55sers/", "/private/", "/var/folders/", "/tmp/"):
        if fragment in text:
            raise SystemExit(f"{label} leaked a local path: {text}")


def test_planner_routes_scheduler_aliases() -> None:
    planner = RuleBasedPlanner()
    for case in [
        "jobs please",
        "scheduled jobs please",
        "show jobs",
        "show scheduled jobs",
        "show latest scheduled jobs",
        "list automations please",
        "automations please",
        "show automations",
        # Real gaps found live 2026-07-09: "show my ..." and the "what jobs are
        # scheduled" question form both fell through to chat.
        "show my scheduled jobs",
        "show my automations",
        "what jobs are scheduled",
    ]:
        actions = planner.plan(case).actions
        if [action.tool_name for action in actions] != ["list_scheduled_jobs"]:
            raise SystemExit(f"planner missed scheduled-job list route for {case!r}: {actions!r}")

    for case in [
        "run jobs now",
        "run due jobs now",
        "run scheduled jobs now",
        "run due scheduled jobs please",
        "run all due jobs please",
    ]:
        actions = planner.plan(case).actions
        if [action.tool_name for action in actions] != ["run_due_jobs"]:
            raise SystemExit(f"planner missed run_due_jobs route for {case!r}: {actions!r}")

    for case, expected_name in [
        ("run weekly review now", "Weekly Review"),
        ("start weekly review now", "Weekly Review"),
        ("trigger weekly review now", "Weekly Review"),
        ("run goal nudge now", "Goal Nudge"),
        ("trigger goal nudge now", "Goal Nudge"),
        ("run job weekly review now", "Weekly Review"),
    ]:
        actions = planner.plan(case).actions
        if [action.tool_name for action in actions] != ["run_job_now"] or actions[0].args != {"name": expected_name}:
            raise SystemExit(f"planner missed run_job_now route for {case!r}: {actions!r}")

    for case, expected_tool, expected_name in [
        ("pause weekly review", "pause_job", "Weekly Review"),
        ("pause job weekly review please", "pause_job", "Weekly Review"),
        ("resume weekly review", "resume_job", "Weekly Review"),
        ("resume job weekly review please", "resume_job", "Weekly Review"),
        ("delete weekly review", "delete_job", "Weekly Review"),
        ("delete job weekly review please", "delete_job", "Weekly Review"),
        # Real bug found live 2026-07-09: "pause the morning brief job" and
        # "run the morning brief job now" fell through to the generic
        # \bmorning brief\b substring catch-all, which ignores the verb
        # entirely -- so a request to PAUSE the job instead composed and
        # displayed today's brief content and never touched the job. Root
        # cause: the job-alias lookup only stripped a leading "job " prefix,
        # not a leading "the " article or trailing " job" suffix, so "the
        # morning brief job" never matched the "morning brief" alias key.
        ("pause the morning brief job", "pause_job", "Morning Brief"),
        ("pause the morning brief", "pause_job", "Morning Brief"),
        ("resume the morning brief job", "resume_job", "Morning Brief"),
        ("delete the weekly review job", "delete_job", "Weekly Review"),
        ("pause the daily brief job", "pause_job", "Daily Brief"),
    ]:
        actions = planner.plan(case).actions
        if [action.tool_name for action in actions] != [expected_tool] or actions[0].args != {"name": expected_name}:
            raise SystemExit(f"planner missed scheduled-job management route for {case!r}: {actions!r}")

    # Companion to the fixes above: "run the X job now" must also resolve
    # through the job-alias normalizer, not just "run X now".
    run_now_actions = planner.plan("run the morning brief job now").actions
    if [a.tool_name for a in run_now_actions] != ["run_job_now"] or run_now_actions[0].args != {"name": "Morning Brief"}:
        raise SystemExit(f"planner missed 'run the X job now' route: {run_now_actions!r}")


def test_scheduled_job_list_invalid_runtime_arguments_stop_before_scheduler_reads_or_handler() -> None:
    invalid_args = (
        {"limit": 1},
        {"include_paused": True},
        [],
        None,
        "scheduled jobs",
    )
    for index, args in enumerate(invalid_args):
        with TemporaryDirectory(prefix="jarvis-scheduled-job-list-arguments-") as temp:
            runtime = make_temp_runtime(Path(temp))
            tool = runtime.registry.get("list_scheduled_jobs")
            contract = tool.argument_contract
            if (
                contract is None
                or contract.version != TOOL_ARGUMENT_CONTRACT_VERSION
                or contract.allow_unknown
                or contract.fields
            ):
                raise SystemExit(
                    f"list_scheduled_jobs should accept only an empty argument object: {contract}"
                )
            calls = [0]

            def counted_handler(value: dict[str, object]):
                calls[0] += 1
                return tool.handler(value)

            runtime.registry._tools["list_scheduled_jobs"] = replace(tool, handler=counted_handler)
            runtime.planner = _StaticPlanner(args)
            with (
                mock.patch.object(
                    runtime.store,
                    "list_jobs",
                    side_effect=AssertionError("invalid scheduled-job-list arguments read scheduler state"),
                ),
                mock.patch.object(
                    runtime.store,
                    "scheduled_delivery_operational_snapshot",
                    side_effect=AssertionError("invalid scheduled-job-list arguments read delivery state"),
                ),
            ):
                result = runtime.handle(
                    "reject invalid scheduled-job-list arguments",
                    request_token=f"scheduled-job-list-invalid-{index}",
                )
            if len(result.tool_results) != 1:
                raise SystemExit(f"invalid scheduled-job-list case {index} lost its result")
            item = result.tool_results[0]
            if (
                item.ok
                or item.metadata.get("failure_kind") != "tool_arguments_invalid"
                or item.metadata.get("handler_invoked") is not False
                or calls[0] != 0
            ):
                raise SystemExit(
                    f"invalid scheduled-job-list case {index} crossed the scheduler boundary: {item}"
                )


def test_scheduler_context_reports_compaction_readiness() -> None:
    with TemporaryDirectory(prefix="jarvis-scheduler-compaction-readiness-") as temp:
        runtime = make_temp_runtime(Path(temp))
        mission_control = Path(temp) / "Vault" / "Jarvis" / "Automations" / "Mission Control.md"
        current_context = Path(temp) / "Vault" / "Jarvis" / "Memory Tree" / "Current Context.md"
        mission_control.parent.mkdir(parents=True, exist_ok=True)
        current_context.parent.mkdir(parents=True, exist_ok=True)
        mission_control.write_text("# Mission Control\n", encoding="utf-8")
        current_context.write_text("# Current Context\n", encoding="utf-8")
        runtime.store.upsert_job("State Snapshot", 360, "state_snapshot", "2099-01-01T00:00:00")

        missing_packet = runtime.registry.get("scheduler_context_refresh_packet").handler({})
        missing_metadata = missing_packet.metadata
        if missing_metadata.get("refresh_state") != "REFRESH_CONVERSATION_COMPACTION_NOT_SCHEDULED":
            raise SystemExit(f"missing compaction should keep scheduler readiness incomplete: {missing_metadata}")
        if missing_metadata.get("next_command") != "schedule assistant basics":
            raise SystemExit(f"missing compaction should point to assistant basics: {missing_metadata}")
        if missing_metadata.get("conversation_compaction_jobs") != 0 or missing_metadata.get("enabled_conversation_compaction_jobs") != 0:
            raise SystemExit(f"missing compaction counters wrong: {missing_metadata}")
        if "enabled Conversation Compaction job" not in missing_metadata.get("missing_readiness", []):
            raise SystemExit(f"missing compaction should appear in missing_readiness: {missing_metadata}")
        if "conversation compaction jobs: 0" not in missing_packet.output:
            raise SystemExit("scheduler context output missed missing compaction counters.")
        assert_operator_limits(missing_metadata, missing_packet.output, "missing compaction scheduler context")
        for key in SAFE_FALSE_FLAGS + ["writes_files", "writes_database", "writes_notes"]:
            if missing_metadata.get(key):
                raise SystemExit(f"missing compaction scheduler context unsafe metadata {key}: {missing_metadata}")

        runtime.store.upsert_job("Conversation Compaction", 1440, "conversation_compaction", "2099-01-01T00:00:00")
        paused = runtime.registry.get("pause_job").handler({"name": "Conversation Compaction"})
        if not paused.ok:
            raise SystemExit(f"pause Conversation Compaction should work for readiness fixture: {paused}")
        paused_packet = runtime.registry.get("scheduler_context_refresh_packet").handler({})
        paused_metadata = paused_packet.metadata
        if paused_metadata.get("refresh_state") != "REFRESH_CONVERSATION_COMPACTION_DISABLED":
            raise SystemExit(f"paused compaction should be reported as disabled: {paused_metadata}")
        if paused_metadata.get("next_command") != "resume job Conversation Compaction":
            raise SystemExit(f"paused compaction should point to resume: {paused_metadata}")
        if paused_metadata.get("disabled_conversation_compaction_jobs") != 1 or paused_metadata.get("enabled_conversation_compaction_jobs") != 0:
            raise SystemExit(f"paused compaction counters wrong: {paused_metadata}")
        if "resume job Conversation Compaction" not in paused_packet.output:
            raise SystemExit("paused compaction output should include the safe resume command.")

        resumed = runtime.registry.get("resume_job").handler({"name": "Conversation Compaction"})
        if not resumed.ok:
            raise SystemExit(f"resume Conversation Compaction should work for readiness fixture: {resumed}")
        ready_packet = runtime.registry.get("scheduler_context_refresh_packet").handler({})
        ready_metadata = ready_packet.metadata
        if ready_metadata.get("refresh_state") != "REFRESH_READY":
            raise SystemExit(f"enabled compaction should restore ready scheduler context: {ready_metadata}")
        if ready_metadata.get("enabled_conversation_compaction_jobs") != 1:
            raise SystemExit(f"ready compaction counter wrong: {ready_metadata}")


def test_scheduler_reports_tolerate_malformed_job_rows() -> None:
    marker = "SCHEDULER_HOSTILE_JOB_ROW_SHOULD_NOT_LEAK"
    with TemporaryDirectory(prefix="jarvis-scheduler-hostile-row-") as temp:
        runtime = make_temp_runtime(Path(temp))
        mission_control = Path(temp) / "Vault" / "Jarvis" / "Automations" / "Mission Control.md"
        current_context = Path(temp) / "Vault" / "Jarvis" / "Memory Tree" / "Current Context.md"
        mission_control.parent.mkdir(parents=True, exist_ok=True)
        current_context.parent.mkdir(parents=True, exist_ok=True)
        mission_control.write_text("# Mission Control\n", encoding="utf-8")
        current_context.write_text("# Current Context\n", encoding="utf-8")
        runtime.store.upsert_job("State Snapshot", 360, "state_snapshot", "2099-01-01T00:00:00")
        runtime.store.upsert_job("Conversation Compaction", 1440, "conversation_compaction", "2099-01-01T00:05:00")
        readable_rows = runtime.store.list_jobs()
        runtime.store.list_jobs = lambda: [HostileJobRow(marker), *readable_rows]  # type: ignore[method-assign]

        packet = runtime.registry.get("scheduler_context_refresh_packet").handler({})
        metadata = packet.metadata
        if not packet.ok:
            raise SystemExit(f"scheduler context should tolerate malformed job rows: {packet}")
        if metadata.get("refresh_state") != "REFRESH_SCHEDULED_JOBS_UNREADABLE":
            raise SystemExit(f"malformed job rows should fail closed to scheduler review: {metadata}")
        if metadata.get("next_command") != "list scheduled jobs":
            raise SystemExit(f"malformed job rows should point to list scheduled jobs: {metadata}")
        if metadata.get("scheduled_jobs") != 3 or metadata.get("readable_scheduled_jobs") != 2:
            raise SystemExit(f"scheduler context should preserve readable job counters: {metadata}")
        if metadata.get("unreadable_scheduled_job_rows") != 1:
            raise SystemExit(f"scheduler context missed unreadable row counter: {metadata}")
        if "State Snapshot" not in packet.output or "Conversation Compaction" not in packet.output:
            raise SystemExit("scheduler context should preserve readable job names behind malformed rows.")
        if "unreadable scheduled job rows: 1" not in packet.output:
            raise SystemExit("scheduler context should report hidden malformed job rows.")
        if marker in packet.output or marker in str(metadata):
            raise SystemExit("scheduler context leaked raw malformed row text.")
        assert_operator_limits(metadata, packet.output, "malformed-row scheduler context")
        for key in SAFE_FALSE_FLAGS + ["writes_files", "writes_database", "writes_notes"]:
            if metadata.get(key):
                raise SystemExit(f"malformed-row scheduler context unsafe metadata {key}: {metadata}")

        listed = runtime.registry.get("list_scheduled_jobs").handler({})
        list_metadata = listed.metadata
        if not listed.ok:
            raise SystemExit(f"list_scheduled_jobs should tolerate malformed job rows: {listed}")
        if list_metadata.get("scheduled_jobs") != 3 or list_metadata.get("readable_scheduled_jobs") != 2:
            raise SystemExit(f"list_scheduled_jobs should preserve readable counters: {list_metadata}")
        if list_metadata.get("unreadable_scheduled_job_rows") != 1:
            raise SystemExit(f"list_scheduled_jobs missed unreadable row counter: {list_metadata}")
        if "State Snapshot" not in listed.output or "Conversation Compaction" not in listed.output:
            raise SystemExit("list_scheduled_jobs should preserve readable jobs behind malformed rows.")
        if "unreadable scheduled job row(s) hidden for safety" not in listed.output:
            raise SystemExit("list_scheduled_jobs should report hidden malformed rows.")
        if marker in listed.output or marker in str(list_metadata):
            raise SystemExit("list_scheduled_jobs leaked raw malformed row text.")
        assert_operator_limits(list_metadata, listed.output, "malformed-row list_scheduled_jobs")
        for key in SAFE_FALSE_FLAGS + ["writes_files", "writes_database", "writes_notes"]:
            if list_metadata.get(key):
                raise SystemExit(f"malformed-row list_scheduled_jobs unsafe metadata {key}: {list_metadata}")


def test_scheduler_display_redacts_hostile_truthiness_job_names() -> None:
    marker = "/\x55sers/example/private/scheduler-hostile-truthiness"
    with TemporaryDirectory(prefix="jarvis-scheduler-hostile-truthiness-") as temp:
        runtime = make_temp_runtime(Path(temp))
        mission_control = Path(temp) / "Vault" / "Jarvis" / "Automations" / "Mission Control.md"
        current_context = Path(temp) / "Vault" / "Jarvis" / "Memory Tree" / "Current Context.md"
        mission_control.parent.mkdir(parents=True, exist_ok=True)
        current_context.parent.mkdir(parents=True, exist_ok=True)
        mission_control.write_text("# Mission Control\n", encoding="utf-8")
        current_context.write_text("# Current Context\n", encoding="utf-8")
        runtime.store.list_jobs = lambda: [  # type: ignore[method-assign]
            {
                "id": 77,
                "name": HostileTruthinessText(marker),
                "job_type": "state_snapshot",
                "enabled": 1,
                "next_run_at": HostileTruthinessText(marker + "/next-run"),
            }
        ]

        listed = runtime.registry.get("list_scheduled_jobs").handler({})
        if not listed.ok:
            raise SystemExit(f"list_scheduled_jobs should tolerate hostile truthiness text: {listed}")
        if "<local-path>" not in listed.output:
            raise SystemExit(f"list_scheduled_jobs should preserve a redacted path marker: {listed.output}")
        assert_no_local_path(listed.output, "hostile-truthiness list_scheduled_jobs output")
        assert_no_local_path(listed.metadata, "hostile-truthiness list_scheduled_jobs metadata")
        if listed.metadata.get("scheduled_jobs") != 1 or listed.metadata.get("readable_scheduled_jobs") != 1:
            raise SystemExit(f"list_scheduled_jobs should keep hostile-text row readable after redaction: {listed.metadata}")
        assert_operator_limits(listed.metadata, listed.output, "hostile-truthiness list_scheduled_jobs")

        packet = runtime.registry.get("scheduler_context_refresh_packet").handler({})
        if not packet.ok:
            raise SystemExit(f"scheduler context should tolerate hostile truthiness text: {packet}")
        if "<local-path>" not in packet.output:
            raise SystemExit(f"scheduler context should preserve a redacted path marker: {packet.output}")
        assert_no_local_path(packet.output, "hostile-truthiness scheduler context output")
        assert_no_local_path(packet.metadata, "hostile-truthiness scheduler context metadata")
        if packet.metadata.get("scheduled_jobs") != 1 or packet.metadata.get("readable_scheduled_jobs") != 1:
            raise SystemExit(f"scheduler context should keep hostile-text row readable after redaction: {packet.metadata}")
        assert_operator_limits(packet.metadata, packet.output, "hostile-truthiness scheduler context")
        for key in SAFE_FALSE_FLAGS + ["writes_files", "writes_database", "writes_notes"]:
            if packet.metadata.get(key):
                raise SystemExit(f"hostile-truthiness scheduler context unsafe metadata {key}: {packet.metadata}")


def test_scheduler_enabled_values_fail_closed() -> None:
    with TemporaryDirectory(prefix="jarvis-scheduler-enabled-fail-closed-") as temp:
        runtime = make_temp_runtime(Path(temp))
        mission_control = Path(temp) / "Vault" / "Jarvis" / "Automations" / "Mission Control.md"
        current_context = Path(temp) / "Vault" / "Jarvis" / "Memory Tree" / "Current Context.md"
        mission_control.parent.mkdir(parents=True, exist_ok=True)
        current_context.parent.mkdir(parents=True, exist_ok=True)
        mission_control.write_text("# Mission Control\n", encoding="utf-8")
        current_context.write_text("# Current Context\n", encoding="utf-8")
        runtime.store.upsert_job("State Snapshot", 360, "state_snapshot", "2099-01-01T00:00:00")
        runtime.store.upsert_job("Conversation Compaction", 1440, "conversation_compaction", "2099-01-01T00:05:00")
        with runtime.store.connect() as conn:
            conn.execute(
                "UPDATE scheduled_jobs SET enabled = 'true' WHERE name IN (?, ?)",
                ("State Snapshot", "Conversation Compaction"),
            )

        packet = runtime.registry.get("scheduler_context_refresh_packet").handler({})
        metadata = packet.metadata
        if metadata.get("refresh_state") != "REFRESH_STATE_SNAPSHOT_DISABLED":
            raise SystemExit(f"scheduler context should fail closed on malformed enabled strings: {metadata}")
        if metadata.get("next_command") != "resume job State Snapshot":
            raise SystemExit(f"malformed enabled rows should point at resume, not list jobs: {metadata}")
        expected_counts = {
            "enabled_jobs": 0,
            "enabled_state_snapshot_jobs": 0,
            "disabled_state_snapshot_jobs": 1,
            "enabled_conversation_compaction_jobs": 0,
            "disabled_conversation_compaction_jobs": 1,
        }
        for key, expected in expected_counts.items():
            if metadata.get(key) != expected:
                raise SystemExit(f"scheduler context malformed-enabled counter {key} expected {expected}: {metadata}")
        if "enabled jobs: 0" not in packet.output or "disabled state snapshot jobs: 1" not in packet.output:
            raise SystemExit(f"scheduler context output missed fail-closed enabled counts: {packet.output}")
        if "State Snapshot [state_snapshot, on]" in packet.output or "Conversation Compaction [conversation_compaction, on]" in packet.output:
            raise SystemExit(f"scheduler context displayed malformed enabled strings as on: {packet.output}")
        assert_operator_limits(metadata, packet.output, "malformed-enabled scheduler context")
        for key in SAFE_FALSE_FLAGS + ["writes_files", "writes_database", "writes_notes"]:
            if metadata.get(key):
                raise SystemExit(f"malformed-enabled scheduler context unsafe metadata {key}: {metadata}")

        listed = runtime.registry.get("list_scheduled_jobs").handler({})
        if "State Snapshot [state_snapshot, off]" not in listed.output:
            raise SystemExit(f"list_scheduled_jobs should display malformed State Snapshot as off: {listed.output}")
        if "Conversation Compaction [conversation_compaction, off]" not in listed.output:
            raise SystemExit(f"list_scheduled_jobs should display malformed Conversation Compaction as off: {listed.output}")
        if "State Snapshot [state_snapshot, on]" in listed.output or "Conversation Compaction [conversation_compaction, on]" in listed.output:
            raise SystemExit(f"list_scheduled_jobs displayed malformed enabled strings as on: {listed.output}")
        assert_operator_limits(listed.metadata, listed.output, "malformed-enabled list_scheduled_jobs")


def test_schedule_assistant_basics_tolerates_malformed_existing_job_rows() -> None:
    marker = "SCHEDULE_BASICS_HOSTILE_ROW_SHOULD_NOT_LEAK"
    with TemporaryDirectory(prefix="jarvis-schedule-basics-hostile-row-") as temp:
        runtime = make_temp_runtime(Path(temp))
        runtime.store.upsert_job("State Snapshot", 360, "state_snapshot", "2099-01-01T00:00:00")
        original_list_jobs = runtime.store.list_jobs
        readable_rows = original_list_jobs()
        runtime.store.list_jobs = lambda: [HostileJobRow(marker), *readable_rows]  # type: ignore[method-assign]

        result = runtime.registry.get("schedule_assistant_basics").handler({})
        metadata = result.metadata
        if not result.ok:
            raise SystemExit(f"schedule assistant basics should tolerate malformed existing rows: {result}")
        if (
            metadata.get("created_jobs") != 6
            or metadata.get("refreshed_jobs") != 0
            or metadata.get("preserved_jobs") != 1
        ):
            raise SystemExit(
                "schedule assistant basics should create only missing defaults and preserve "
                f"readable rows: {metadata}"
            )
        if metadata.get("unreadable_scheduled_job_rows") != 1:
            raise SystemExit(f"schedule assistant basics missed unreadable row metadata: {metadata}")
        if "State Snapshot" not in metadata.get("existing_job_names", []):
            raise SystemExit(f"schedule assistant basics should preserve safe existing job names: {metadata}")
        if "unreadable scheduled job row(s) ignored while creating missing defaults" not in result.output:
            raise SystemExit(f"schedule assistant basics should report hidden unreadable rows: {result.output}")
        if marker in result.output or marker in str(metadata):
            raise SystemExit("schedule assistant basics leaked a raw malformed scheduled-job row.")
        assert_operator_limits(metadata, result.output, "malformed-row schedule assistant basics")
        for key in SAFE_FALSE_FLAGS:
            if metadata.get(key):
                raise SystemExit(f"malformed-row schedule assistant basics unsafe metadata {key}: {metadata}")

        jobs = original_list_jobs()
        default_names = {
            "Daily Brief",
            "Inbox Ingest",
            "Recent File Digest",
            "Goal Nudge",
            "Weekly Review",
            "State Snapshot",
            "Conversation Compaction",
        }
        if {row["name"] for row in jobs} != default_names:
            raise SystemExit(f"schedule assistant basics should leave exactly the default jobs in storage: {jobs}")


def test_schedule_assistant_basics_preserves_operator_state() -> None:
    with TemporaryDirectory(prefix="jarvis-schedule-basics-preserve-") as temp:
        runtime = make_temp_runtime(Path(temp))
        runtime.store.upsert_job(
            "Daily Brief",
            17,
            "custom_daily_brief",
            "2099-03-01T01:02:03",
        )
        runtime.store.upsert_job(
            "state snapshot",
            23,
            "custom_state_snapshot",
            "2099-04-05T06:07:08",
        )
        paused = runtime.registry.get("pause_job").handler({"name": "state snapshot"})
        if not paused.ok:
            raise SystemExit(f"preservation fixture could not pause its custom job: {paused}")
        before = {
            int(row["id"]): dict(row)
            for row in runtime.store.list_jobs()
        }

        result = runtime.registry.get("schedule_assistant_basics").handler({})
        metadata = result.metadata
        if not result.ok:
            raise SystemExit(f"schedule assistant basics should preserve existing jobs: {result}")
        expected_counts = {
            "created_jobs": 5,
            "refreshed_jobs": 0,
            "preserved_jobs": 2,
            "paused_preserved_jobs": 1,
        }
        for key, expected in expected_counts.items():
            if metadata.get(key) != expected:
                raise SystemExit(f"schedule assistant basics {key} expected {expected}: {metadata}")
        if "(existing; preserved)" not in result.output:
            raise SystemExit("enabled custom schedule was not reported as preserved")
        if "Daily Brief every 17 minutes (existing; preserved)" not in result.output:
            raise SystemExit("preserved custom schedule displayed the bootstrap interval")
        if "(paused; preserved; use resume job State Snapshot)" not in result.output:
            raise SystemExit("paused custom schedule was not reported with its resume command")
        if "State Snapshot every 23 minutes (paused; preserved" not in result.output:
            raise SystemExit("paused custom schedule displayed the bootstrap interval")

        after_rows = runtime.store.list_jobs()
        after = {int(row["id"]): dict(row) for row in after_rows}
        for job_id, snapshot in before.items():
            if after.get(job_id) != snapshot:
                raise SystemExit(
                    "schedule assistant basics changed existing operator state: "
                    f"before={snapshot}, after={after.get(job_id)}"
                )
        if len(after_rows) != 7:
            raise SystemExit("schedule assistant basics did not create exactly the missing defaults")
        if sum(str(row["name"]).casefold() == "state snapshot" for row in after_rows) != 1:
            raise SystemExit("case-insensitive default identity created a duplicate State Snapshot")
        if metadata.get("writes_files") is not False or metadata.get("writes_database") is not True:
            raise SystemExit("partially creating defaults reported incorrect write effects")

        repeated = runtime.registry.get("schedule_assistant_basics").handler({})
        if (
            repeated.metadata.get("created_jobs") != 0
            or repeated.metadata.get("writes_database") is not False
            or repeated.metadata.get("writes_files") is not False
        ):
            raise SystemExit("no-op default bootstrap claimed writes")


def test_create_job_if_missing_is_concurrent_and_case_insensitive() -> None:
    with TemporaryDirectory(prefix="jarvis-schedule-create-once-") as temp:
        runtime = make_temp_runtime(Path(temp))
        barrier = Barrier(2)
        results: Queue[object] = Queue()

        def create(name: str) -> None:
            try:
                barrier.wait(timeout=5)
                results.put(
                    runtime.store.create_job_if_missing(
                        name,
                        60,
                        "create_once_test",
                        "2099-01-01T00:00:00",
                    )
                )
            except BaseException as exc:
                results.put(exc)

        threads = [
            Thread(target=create, args=("Bootstrap Race",)),
            Thread(target=create, args=("bootstrap race",)),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)
            if thread.is_alive():
                raise SystemExit("concurrent create-if-missing worker did not finish")

        outcomes = [results.get_nowait(), results.get_nowait()]
        if any(isinstance(outcome, BaseException) for outcome in outcomes):
            raise SystemExit(f"concurrent create-if-missing failed: {outcomes}")
        ids = {int(outcome[0]["id"]) for outcome in outcomes}  # type: ignore[index]
        created = [bool(outcome[1]) for outcome in outcomes]  # type: ignore[index]
        if len(ids) != 1 or sum(created) != 1:
            raise SystemExit(f"concurrent create-if-missing did not converge: {outcomes}")
        matching = [
            row
            for row in runtime.store.list_jobs()
            if str(row["name"]).casefold() == "bootstrap race"
        ]
        if len(matching) != 1:
            raise SystemExit("case-insensitive concurrent create produced duplicate jobs")

        first, first_created = runtime.store.create_job_if_missing(
            "Éclair",
            60,
            "unicode_create_once_test",
            "2099-01-01T00:00:00",
        )
        second, second_created = runtime.store.create_job_if_missing(
            "éclair",
            60,
            "unicode_create_once_test",
            "2099-01-01T00:00:00",
        )
        if not first_created or second_created or int(first["id"]) != int(second["id"]):
            raise SystemExit("Unicode-casefold create-if-missing identity did not converge")


def test_schedule_assistant_basics_rejects_unknown_arguments() -> None:
    with TemporaryDirectory(prefix="jarvis-schedule-basics-contract-") as temp:
        runtime = make_temp_runtime(Path(temp))
        result = runtime.executor.execute(
            PlannedAction(
                "schedule_assistant_basics",
                {"unexpected": "PRIVATE-SCHEDULER-MARKER"},
                "typed scheduler smoke",
            )
        )
        if (
            result.ok
            or result.metadata.get("failure_kind") != "tool_arguments_invalid"
            or result.metadata.get("handler_invoked") is not False
        ):
            raise SystemExit(f"schedule assistant basics accepted unknown arguments: {result}")
        if runtime.store.list_jobs():
            raise SystemExit("invalid schedule assistant basics arguments created jobs")
        if "PRIVATE-SCHEDULER-MARKER" in result.output or "PRIVATE-SCHEDULER-MARKER" in str(result.metadata):
            raise SystemExit("typed scheduler rejection leaked an unknown argument value")


def test_schedule_assistant_basics_preserves_malformed_matching_row() -> None:
    with TemporaryDirectory(prefix="jarvis-schedule-basics-malformed-match-") as temp:
        runtime = make_temp_runtime(Path(temp))
        job_id = runtime.store.upsert_job(
            "Daily Brief",
            17,
            "daily_brief",
            "2099-01-01T00:00:00",
        )
        with runtime.store.connect() as conn:
            conn.execute(
                """
                UPDATE scheduled_jobs
                SET interval_minutes = -9, job_type = '', next_run_at = 'not-a-time',
                    enabled = 2, schedule_revision = -1, metadata = '[]'
                WHERE id = ?
                """,
                (job_id,),
            )
        before = dict(next(row for row in runtime.store.list_jobs() if row["id"] == job_id))

        result = runtime.registry.get("schedule_assistant_basics").handler({})
        if not result.ok:
            raise SystemExit(f"malformed matching row should be preserved for review: {result}")
        if (
            result.metadata.get("created_jobs") != 6
            or result.metadata.get("preserved_jobs") != 1
            or result.metadata.get("unreadable_preserved_jobs") != 1
        ):
            raise SystemExit(f"malformed preserved-row counts were wrong: {result.metadata}")
        if "Daily Brief (existing state unreadable; preserved; inspect list scheduled jobs)" not in result.output:
            raise SystemExit("malformed preserved row was presented as healthy")
        after = dict(next(row for row in runtime.store.list_jobs() if row["id"] == job_id))
        if after != before:
            raise SystemExit("malformed matching row was silently repaired or changed")


def _capture_due_run(scheduler: Scheduler, results: Queue[object]) -> None:
    try:
        results.put(scheduler.run_due_jobs())
    except BaseException as exc:
        results.put(exc)


def test_concurrent_schedulers_claim_due_job_once() -> None:
    with TemporaryDirectory(prefix="jarvis-scheduler-concurrent-claim-") as temp:
        runtime = make_temp_runtime(Path(temp))
        due_at = iso(datetime.now() - timedelta(minutes=2))
        runtime.store.upsert_job("Concurrent Claim", 60, "concurrent_claim_test", due_at)
        schedulers = [
            Scheduler(runtime.store, runtime.vault, runtime.config),
            Scheduler(runtime.store, runtime.vault, runtime.config),
        ]
        start = Barrier(3)
        handler_entered = Event()
        release_handler = Event()
        invocation_lock = Lock()
        invocation_count = 0
        results: Queue[object] = Queue()

        def handler(job_type: str, *, row=None, now=None, lease_authority=None) -> str:
            nonlocal invocation_count
            if job_type != "concurrent_claim_test":
                raise AssertionError(f"unexpected mocked job type: {job_type}")
            with invocation_lock:
                invocation_count += 1
            handler_entered.set()
            if not release_handler.wait(timeout=10):
                raise AssertionError("timed out waiting to release mocked scheduler handler")
            return "claimed once"

        for scheduler in schedulers:
            scheduler._run_job_type = handler  # type: ignore[method-assign]

        def run_scheduler(scheduler: Scheduler) -> None:
            try:
                start.wait(timeout=10)
                results.put(scheduler.run_due_jobs())
            except BaseException as exc:
                results.put(exc)

        threads = [Thread(target=run_scheduler, args=(scheduler,)) for scheduler in schedulers]
        for thread in threads:
            thread.start()
        start.wait(timeout=10)
        if not handler_entered.wait(timeout=10):
            raise SystemExit("concurrent scheduler test never entered the claimed handler")

        while_claimed = [row for row in runtime.store.list_jobs() if row["name"] == "Concurrent Claim"][0]
        if while_claimed["next_run_at"] != due_at:
            raise SystemExit(f"claim should preserve the row's original timing: {dict(while_claimed)}")
        if not while_claimed["lease_token"] or not while_claimed["lease_expires_at"]:
            raise SystemExit(f"claimed row should persist a bounded lease: {dict(while_claimed)}")

        first_result = results.get(timeout=10)
        if first_result != "No jobs due.":
            raise SystemExit(f"concurrent scheduler should observe the active lease: {first_result!r}")
        release_handler.set()
        second_result = results.get(timeout=10)
        for thread in threads:
            thread.join(timeout=10)
        if any(thread.is_alive() for thread in threads):
            raise SystemExit("concurrent scheduler test left a worker thread running")
        if isinstance(second_result, BaseException):
            raise SystemExit(f"claimed scheduler raised unexpectedly: {second_result!r}")
        if invocation_count != 1 or "Ran Concurrent Claim" not in str(second_result):
            raise SystemExit(
                f"two synchronized schedulers should invoke one handler exactly once: "
                f"count={invocation_count}, output={second_result!r}"
            )
        completed = [row for row in runtime.store.list_jobs() if row["name"] == "Concurrent Claim"][0]
        if completed["lease_token"] is not None or completed["lease_expires_at"] is not None:
            raise SystemExit(f"successful scheduler completion should clear its lease: {dict(completed)}")


def test_later_due_job_is_not_preclaimed_behind_slow_handler() -> None:
    with TemporaryDirectory(prefix="jarvis-scheduler-per-job-claim-") as temp:
        runtime = make_temp_runtime(Path(temp))
        now = datetime.now()
        runtime.store.upsert_job("First Slow", 60, "first_slow_test", iso(now - timedelta(minutes=2)))
        runtime.store.upsert_job("Later Due", 60, "later_due_test", iso(now - timedelta(minutes=1)))
        schedulers = [
            Scheduler(
                runtime.store,
                runtime.vault,
                runtime.config,
                job_lease_seconds=0.15,
                job_heartbeat_seconds=0.03,
            )
            for _ in range(2)
        ]
        first_entered = Event()
        release_first = Event()
        invocation_lock = Lock()
        invocations: list[str] = []
        first_result: Queue[object] = Queue()

        def handler(job_type: str, *, row=None, now=None, lease_authority=None) -> str:
            with invocation_lock:
                invocations.append(job_type)
            if job_type == "first_slow_test":
                first_entered.set()
                if not release_first.wait(timeout=5):
                    raise AssertionError("timed out waiting to release first due handler")
            return f"completed {job_type}"

        for scheduler in schedulers:
            scheduler._run_job_type = handler  # type: ignore[method-assign]

        thread = Thread(target=_capture_due_run, args=(schedulers[0], first_result))
        thread.start()
        try:
            if not first_entered.wait(timeout=5):
                raise SystemExit("slow first due job never entered its handler")
            later = {row["name"]: row for row in runtime.store.list_jobs()}["Later Due"]
            if later["lease_token"] is not None or later["lease_expires_at"] is not None:
                raise SystemExit(f"later due row must remain unclaimed until execution starts: {dict(later)}")

            time.sleep(0.22)
            later_after_base_lease = {row["name"]: row for row in runtime.store.list_jobs()}["Later Due"]
            if later_after_base_lease["lease_token"] is not None:
                raise SystemExit(
                    f"later due row acquired a stale batch lease behind the slow handler: {dict(later_after_base_lease)}"
                )

            second_output = schedulers[1].run_due_jobs()
            if "Ran Later Due" not in second_output:
                raise SystemExit(f"another scheduler should claim the untouched later row once: {second_output!r}")
        finally:
            release_first.set()
            thread.join(timeout=5)
        if thread.is_alive():
            raise SystemExit("per-job claim test left the first scheduler thread running")
        first_output = first_result.get(timeout=5)
        if isinstance(first_output, BaseException):
            raise SystemExit(f"first scheduler raised unexpectedly: {first_output!r}")
        with invocation_lock:
            observed = list(invocations)
        if observed.count("first_slow_test") != 1 or observed.count("later_due_test") != 1:
            raise SystemExit(f"per-job claiming should execute each due handler once: {observed}")


def test_heartbeat_keeps_long_handler_exclusively_owned() -> None:
    with TemporaryDirectory(prefix="jarvis-scheduler-heartbeat-") as temp:
        runtime = make_temp_runtime(Path(temp))
        runtime.store.upsert_job(
            "Heartbeat Guard",
            60,
            "heartbeat_guard_test",
            iso(datetime.now() - timedelta(minutes=1)),
        )
        schedulers = [
            Scheduler(
                runtime.store,
                runtime.vault,
                runtime.config,
                job_lease_seconds=0.2,
                job_heartbeat_seconds=0.04,
            )
            for _ in range(2)
        ]
        handler_entered = Event()
        release_handler = Event()
        invocation_lock = Lock()
        invocation_count = 0
        owner_result: Queue[object] = Queue()

        def handler(job_type: str, *, row=None, now=None, lease_authority=None) -> str:
            nonlocal invocation_count
            with invocation_lock:
                invocation_count += 1
                invocation_number = invocation_count
            if invocation_number == 1:
                handler_entered.set()
                if not release_handler.wait(timeout=5):
                    raise AssertionError("timed out waiting to release heartbeat handler")
            return "heartbeat protected"

        for scheduler in schedulers:
            scheduler._run_job_type = handler  # type: ignore[method-assign]

        thread = Thread(target=_capture_due_run, args=(schedulers[0], owner_result))
        thread.start()
        try:
            if not handler_entered.wait(timeout=5):
                raise SystemExit("heartbeat test never entered the owned handler")
            initial = {row["name"]: row for row in runtime.store.list_jobs()}["Heartbeat Guard"]
            initial_expiry = datetime.fromisoformat(str(initial["lease_expires_at"]))
            deadline = time.monotonic() + 2
            renewed = None
            while time.monotonic() < deadline:
                current = {row["name"]: row for row in runtime.store.list_jobs()}["Heartbeat Guard"]
                current_expiry = datetime.fromisoformat(str(current["lease_expires_at"]))
                if datetime.now() > initial_expiry and current_expiry > datetime.now():
                    renewed = current
                    break
                time.sleep(0.01)
            if renewed is None:
                raise SystemExit("heartbeat did not renew the claim beyond its base lease")

            contender_output = schedulers[1].run_due_jobs()
            if contender_output != "No jobs due.":
                raise SystemExit(f"renewed handler lease should reject a competing scheduler: {contender_output!r}")
            with invocation_lock:
                if invocation_count != 1:
                    raise SystemExit(f"long handler lost exclusive ownership: count={invocation_count}")
        finally:
            release_handler.set()
            thread.join(timeout=5)
        if thread.is_alive():
            raise SystemExit("heartbeat test left its owner scheduler thread running")
        completed = owner_result.get(timeout=5)
        if isinstance(completed, BaseException) or "Ran Heartbeat Guard" not in str(completed):
            raise SystemExit(f"heartbeat-protected scheduler did not complete normally: {completed!r}")


def test_effect_boundary_atomically_renews_and_fails_closed() -> None:
    with TemporaryDirectory(prefix="jarvis-scheduler-durable-boundary-") as temp:
        runtime = make_temp_runtime(Path(temp))
        scheduler = Scheduler(runtime.store, runtime.vault, runtime.config, job_lease_seconds=5)

        def claimed_heartbeat(name: str):
            job_id = runtime.store.upsert_job(
                name,
                60,
                "daily_brief",
                iso(datetime.now() - timedelta(minutes=1)),
            )
            token = name.casefold().replace(" ", "-")
            claimed = runtime.store.claim_job(job_id, token, scheduler.job_lease_seconds)
            if claimed is None:
                raise SystemExit(f"{name} boundary fixture could not claim its scheduled job")
            return job_id, token, scheduler._claim_heartbeat(claimed, token)

        job_id, token, heartbeat = claimed_heartbeat("Boundary Renewal")
        with runtime.store.connect() as conn:
            conn.execute(
                "UPDATE scheduled_jobs SET lease_expires_at = ? WHERE id = ?",
                (iso(datetime.now() + timedelta(seconds=1)), job_id),
            )
        original_renew = runtime.store.renew_job_claim
        with mock.patch.object(runtime.store, "renew_job_claim", wraps=original_renew) as renew_spy:
            heartbeat.require_authority()
        renew_spy.assert_called_once_with(job_id, token, scheduler.job_lease_seconds)
        renewed_row = next(row for row in runtime.store.list_jobs() if int(row["id"]) == job_id)
        renewed_until = datetime.fromisoformat(str(renewed_row["lease_expires_at"]))
        if renewed_until <= datetime.now() + timedelta(seconds=4):
            raise SystemExit(f"effect boundary did not extend a fresh durable lease: {dict(renewed_row)}")
        if heartbeat._remaining_confirmed_authority() <= 4:  # type: ignore[attr-defined]
            raise SystemExit("effect boundary did not refresh its conservative monotonic deadline")

        def assert_production_write_blocked(authority, label: str) -> None:
            writes: list[str] = []
            with mock.patch.object(
                runtime.vault,
                "append_daily",
                side_effect=lambda *_args, **_kwargs: writes.append(label),
            ):
                try:
                    scheduler._run_job_type("daily_brief", lease_authority=authority)
                except RuntimeError:
                    pass
                else:
                    raise SystemExit(f"{label} did not fail closed at the effect boundary")
            if writes:
                raise SystemExit(f"{label} reached the production writer: {writes}")

        expired_id, _, expired = claimed_heartbeat("Boundary Expiry")
        with runtime.store.connect() as conn:
            conn.execute(
                "UPDATE scheduled_jobs SET lease_expires_at = ? WHERE id = ?",
                (iso(datetime.now() - timedelta(seconds=1)), expired_id),
            )
        assert_production_write_blocked(expired.require_authority, "durable expiry")

        replaced_id, _, replaced = claimed_heartbeat("Boundary Token Loss")
        with runtime.store.connect() as conn:
            conn.execute(
                "UPDATE scheduled_jobs SET lease_token = ? WHERE id = ?",
                ("replacement-token", replaced_id),
            )
        assert_production_write_blocked(replaced.require_authority, "durable token loss")

        _, _, unavailable = claimed_heartbeat("Boundary Storage Error")
        with mock.patch.object(
            runtime.store,
            "renew_job_claim",
            side_effect=RuntimeError("durable store unavailable"),
        ):
            assert_production_write_blocked(unavailable.require_authority, "durable storage exception")


def test_production_handlers_reject_lost_authority_at_first_write() -> None:
    with TemporaryDirectory(prefix="jarvis-scheduler-production-boundaries-") as temp:
        root = Path(temp)
        runtime = make_temp_runtime(root)
        watched = root / "Watched"
        watched.mkdir(exist_ok=True)
        (watched / "recent.txt").write_text("production boundary", encoding="utf-8")
        (runtime.vault.root_path / "Inbox.md").write_text("# Inbox\n\n- production inbox boundary\n", encoding="utf-8")
        scheduler = Scheduler(runtime.store, runtime.vault, runtime.config)

        cases = (
            ("daily_brief", runtime.vault, "append_daily"),
            ("goal_nudge", runtime.vault, "append_daily"),
            ("weekly_review", runtime.vault, "write_reflection"),
            ("inbox_ingest", runtime.store, "add_memory_if_source_new"),
            ("recent_file_digest", runtime.vault, "append_daily"),
        )
        for job_type, writer_owner, writer_name in cases:
            guard_calls = 0
            writes: list[str] = []

            def authority() -> None:
                nonlocal guard_calls
                guard_calls += 1
                if guard_calls > 1:
                    raise RuntimeError("durable lease unavailable at production write boundary")

            with mock.patch.object(
                writer_owner,
                writer_name,
                side_effect=lambda *_args, **_kwargs: writes.append(job_type),
            ):
                try:
                    scheduler._run_job_type(job_type, lease_authority=authority)
                except RuntimeError:
                    pass
                else:
                    raise SystemExit(f"{job_type} did not reject lost authority before its first write")
            if writes or guard_calls != 2:
                raise SystemExit(
                    f"{job_type} crossed its first production write boundary after authority loss: "
                    f"writes={writes}, guard_calls={guard_calls}"
                )


def test_inbox_mirror_boundaries_abort_without_unsafe_continuation() -> None:
    with TemporaryDirectory(prefix="jarvis-scheduler-inbox-boundaries-") as temp:
        runtime = make_temp_runtime(Path(temp))
        inbox = runtime.vault.root_path / "Inbox.md"
        inbox.write_text("# Inbox\n\n- first pending mirror\n\n- must not continue\n", encoding="utf-8")
        seeded = runtime.store.add_memory_if_source_new
        from jarvis_v2.memory.store import MemoryRecord

        inserted, memory_id = seeded(
            MemoryRecord("inbox", "first pending mirror", "first pending mirror", "obsidian-inbox", 0.9),
            "obsidian-inbox:"
            + hashlib.sha256("first pending mirror".encode("utf-8")).hexdigest(),
            "obsidian-inbox",
        )
        if not inserted or memory_id is None:
            raise SystemExit("inbox production-boundary fixture was not seeded")
        scheduler = Scheduler(runtime.store, runtime.vault, runtime.config)

        for rejected_call, expected_file_writes in ((3, 0), (4, 1)):
            guard_calls = 0
            file_writes = 0

            def authority() -> None:
                nonlocal guard_calls
                guard_calls += 1
                if guard_calls == rejected_call:
                    raise RuntimeError("inbox durable authority lost")

            def write_memory_projection_with_evidence(*_args, **kwargs):
                nonlocal file_writes
                file_writes += 1
                path = (
                    runtime.vault.root_path
                    / "Memory Tree"
                    / "Records"
                    / f"{kwargs['memory_id']:06d} [{kwargs['store_identity']}].md"
                )
                return path, hashlib.sha256(b"mock projection").hexdigest()

            with mock.patch.object(
                runtime.vault,
                "write_memory_projection_with_evidence",
                side_effect=write_memory_projection_with_evidence,
            ):
                try:
                    scheduler._run_job_type("inbox_ingest", lease_authority=authority)
                except RuntimeError:
                    pass
                else:
                    raise SystemExit(f"inbox boundary call {rejected_call} did not fail closed")
            with runtime.store.connect() as conn:
                rows = list(conn.execute("SELECT source_key, mirror_state FROM ingested_sources ORDER BY source_key"))
            if file_writes != expected_file_writes or len(rows) != 1 or rows[0]["mirror_state"] != "pending":
                raise SystemExit(
                    "inbox authority loss wrote or continued unsafely: "
                    f"boundary={rejected_call}, file_writes={file_writes}, rows={[dict(row) for row in rows]}"
                )


def test_transient_heartbeat_error_recovers_before_production_write() -> None:
    with TemporaryDirectory(prefix="jarvis-scheduler-heartbeat-transient-") as temp:
        runtime = make_temp_runtime(Path(temp))
        runtime.store.upsert_job("Transient Daily Brief", 60, "daily_brief", iso(datetime.now() - timedelta(minutes=1)))
        scheduler = Scheduler(runtime.store, runtime.vault, runtime.config, job_lease_seconds=0.25, job_heartbeat_seconds=0.03)
        read_entered = Event()
        release_read = Event()
        transient_seen = Event()
        original_snapshot = runtime.store.read_daily_brief_snapshot
        original_renew = runtime.store.renew_job_claim
        failed_once = False
        writes = 0

        def delayed_snapshot(*args, **kwargs):
            read_entered.set()
            if not release_read.wait(timeout=5):
                raise AssertionError("transient production reader was not released")
            return original_snapshot(*args, **kwargs)

        def flaky_renew(*args, **kwargs):
            nonlocal failed_once
            if current_thread().name.startswith("jarvis-job-lease-") and read_entered.is_set() and not failed_once:
                failed_once = True
                transient_seen.set()
                raise RuntimeError("transient durable renewal outage")
            return original_renew(*args, **kwargs)

        def append_scheduled_daily(*args, **_kwargs):
            nonlocal writes
            writes += 1
            return runtime.vault.root_path / "Daily" / "transient.md", True, str(args[3])

        result: Queue[object] = Queue()
        runtime.store.renew_job_claim = flaky_renew  # type: ignore[method-assign]
        with mock.patch.object(
            runtime.store,
            "read_daily_brief_snapshot",
            side_effect=delayed_snapshot,
        ), mock.patch.object(
            runtime.vault,
            "append_scheduled_daily_once_for_date",
            side_effect=append_scheduled_daily,
        ):
            thread = Thread(target=_capture_due_run, args=(scheduler, result))
            thread.start()
            try:
                if not read_entered.wait(timeout=5) or not transient_seen.wait(timeout=5):
                    raise SystemExit("transient production renewal outage was not exercised")
            finally:
                release_read.set()
                thread.join(timeout=5)
                runtime.store.renew_job_claim = original_renew  # type: ignore[method-assign]
        output = result.get(timeout=5)
        if thread.is_alive() or isinstance(output, BaseException) or "Ran Transient Daily Brief" not in str(output):
            raise SystemExit(f"transient production boundary did not recover: {output!r}")
        if writes != 1:
            raise SystemExit(f"transient recovery should cross the production write boundary once: {writes}")


def test_sustained_heartbeat_errors_fence_production_writer_without_overlap() -> None:
    with TemporaryDirectory(prefix="jarvis-scheduler-heartbeat-sustained-") as temp:
        runtime = make_temp_runtime(Path(temp))
        runtime.store.upsert_job("Sustained Daily Brief", 60, "daily_brief", iso(datetime.now() - timedelta(minutes=1)))
        owner = Scheduler(runtime.store, runtime.vault, runtime.config, job_lease_seconds=0.2, job_heartbeat_seconds=0.03)
        contender = Scheduler(runtime.store, runtime.vault, runtime.config, job_lease_seconds=0.2, job_heartbeat_seconds=0.03)
        original_snapshot = runtime.store.read_daily_brief_snapshot
        original_renew = runtime.store.renew_job_claim
        original_claim_heartbeat = owner._claim_heartbeat
        owner_read_entered = Event()
        release_owner_read = Event()
        contender_writer_entered = Event()
        release_contender_writer = Event()
        captured_heartbeats: list[object] = []
        owner_token: str | None = None
        read_calls = 0
        active_writes = 0
        maximum_active_writes = 0
        completed_writes = 0
        state_lock = Lock()

        def capture_heartbeat(row, lease_token):
            heartbeat = original_claim_heartbeat(row, lease_token)
            captured_heartbeats.append(heartbeat)
            return heartbeat

        def unavailable_renew(job_id, lease_token, lease_seconds, **kwargs):
            nonlocal owner_token
            if owner_token is None:
                owner_token = lease_token
            if lease_token == owner_token and owner_read_entered.is_set():
                raise RuntimeError("sustained durable renewal outage")
            return original_renew(job_id, lease_token, lease_seconds, **kwargs)

        def delayed_first_snapshot(*args, **kwargs):
            nonlocal read_calls
            with state_lock:
                read_calls += 1
                call_number = read_calls
            if call_number == 1:
                owner_read_entered.set()
                if not release_owner_read.wait(timeout=5):
                    raise AssertionError("stale production reader was not released")
            return original_snapshot(*args, **kwargs)

        def append_scheduled_daily(*args, **_kwargs):
            nonlocal active_writes, maximum_active_writes, completed_writes
            with state_lock:
                active_writes += 1
                maximum_active_writes = max(maximum_active_writes, active_writes)
            contender_writer_entered.set()
            try:
                if not release_contender_writer.wait(timeout=5):
                    raise AssertionError("replacement production writer was not released")
                with state_lock:
                    completed_writes += 1
            finally:
                with state_lock:
                    active_writes -= 1
            return runtime.vault.root_path / "Daily" / "replacement.md", True, str(args[3])

        runtime.store.renew_job_claim = unavailable_renew  # type: ignore[method-assign]
        owner._claim_heartbeat = capture_heartbeat  # type: ignore[method-assign]
        owner_result: Queue[object] = Queue()
        contender_result: Queue[object] = Queue()
        owner_thread = Thread(target=_capture_due_run, args=(owner, owner_result))
        contender_thread: Thread | None = None
        with mock.patch.object(
            runtime.store,
            "read_daily_brief_snapshot",
            side_effect=delayed_first_snapshot,
        ), mock.patch.object(
            runtime.vault,
            "append_scheduled_daily_once_for_date",
            side_effect=append_scheduled_daily,
        ):
            owner_thread.start()
            try:
                if not owner_read_entered.wait(timeout=5) or not captured_heartbeats:
                    raise SystemExit("sustained outage never entered the production daily brief")
                heartbeat = captured_heartbeats[0]
                deadline = time.monotonic() + 5
                while not heartbeat.ownership_lost and time.monotonic() < deadline:  # type: ignore[attr-defined]
                    time.sleep(0.005)
                if not heartbeat.ownership_lost:  # type: ignore[attr-defined]
                    raise SystemExit("sustained outage did not expire durable production authority")
                contender_thread = Thread(target=_capture_due_run, args=(contender, contender_result))
                contender_thread.start()
                if not contender_writer_entered.wait(timeout=5):
                    raise SystemExit("replacement scheduler did not reach the production writer")
                release_owner_read.set()
            finally:
                release_owner_read.set()
                release_contender_writer.set()
                owner_thread.join(timeout=5)
                if contender_thread is not None:
                    contender_thread.join(timeout=5)
                runtime.store.renew_job_claim = original_renew  # type: ignore[method-assign]
                owner._claim_heartbeat = original_claim_heartbeat  # type: ignore[method-assign]
        owner_output = owner_result.get(timeout=5)
        contender_output = contender_result.get(timeout=5)
        if owner_thread.is_alive() or contender_thread is None or contender_thread.is_alive():
            raise SystemExit("sustained production fencing left a scheduler thread running")
        if isinstance(owner_output, BaseException) or "Skipped stale claim for Sustained Daily Brief" not in str(owner_output):
            raise SystemExit(f"stale production owner was not fenced cleanly: {owner_output!r}")
        if isinstance(contender_output, BaseException) or "Ran Sustained Daily Brief" not in str(contender_output):
            raise SystemExit(f"replacement production owner did not finish: {contender_output!r}")
        with state_lock:
            if completed_writes != 1 or maximum_active_writes != 1 or active_writes != 0:
                raise SystemExit(
                    "sustained outage allowed stale or overlapping production writes: "
                    f"completed={completed_writes}, maximum={maximum_active_writes}, active={active_writes}"
                )


def test_run_job_now_refuses_due_run_overlap() -> None:
    with TemporaryDirectory(prefix="jarvis-scheduler-manual-overlap-") as temp:
        runtime = make_temp_runtime(Path(temp))
        runtime.store.upsert_job(
            "Manual Overlap",
            60,
            "manual_overlap_test",
            iso(datetime.now() - timedelta(minutes=1)),
        )
        due_scheduler = Scheduler(runtime.store, runtime.vault, runtime.config)
        manual_scheduler = Scheduler(runtime.store, runtime.vault, runtime.config)
        handler_entered = Event()
        release_handler = Event()
        invocation_lock = Lock()
        invocation_count = 0
        due_result: Queue[object] = Queue()

        def handler(job_type: str, *, row=None, now=None, lease_authority=None) -> str:
            nonlocal invocation_count
            with invocation_lock:
                invocation_count += 1
            handler_entered.set()
            if not release_handler.wait(timeout=5):
                raise AssertionError("timed out waiting to release overlap handler")
            return "overlap protected"

        due_scheduler._run_job_type = handler  # type: ignore[method-assign]
        manual_scheduler._run_job_type = handler  # type: ignore[method-assign]
        thread = Thread(target=_capture_due_run, args=(due_scheduler, due_result))
        thread.start()
        try:
            if not handler_entered.wait(timeout=5):
                raise SystemExit("overlap test never entered the due handler")
            manual_output = manual_scheduler.run_job_now("Manual Overlap")
            if "already running" not in manual_output or "run skipped" not in manual_output:
                raise SystemExit(f"run_job_now should refuse an active due claim: {manual_output!r}")
            with invocation_lock:
                if invocation_count != 1:
                    raise SystemExit(f"due/manual overlap invoked the handler more than once: {invocation_count}")
        finally:
            release_handler.set()
            thread.join(timeout=5)
        if thread.is_alive():
            raise SystemExit("run_job_now overlap test left the due scheduler thread running")
        output = due_result.get(timeout=5)
        if isinstance(output, BaseException) or "Ran Manual Overlap" not in str(output):
            raise SystemExit(f"owned due run did not complete after manual refusal: {output!r}")


def test_upsert_during_active_handler_preserves_lease() -> None:
    with TemporaryDirectory(prefix="jarvis-scheduler-active-upsert-") as temp:
        runtime = make_temp_runtime(Path(temp))
        due_at = iso(datetime.now() - timedelta(minutes=1))
        rescheduled_at = "2099-07-11T04:30:00"
        job_id = runtime.store.upsert_job("Active Refresh", 60, "active_refresh_test", due_at)
        runtime.store.merge_job_metadata(job_id, {"unrelated": "preserved"})
        owner = Scheduler(runtime.store, runtime.vault, runtime.config)
        contender = Scheduler(runtime.store, runtime.vault, runtime.config)
        handler_entered = Event()
        release_handler = Event()
        invocation_lock = Lock()
        invocation_count = 0
        owner_result: Queue[object] = Queue()

        def handler(job_type: str, *, row=None, now=None, lease_authority=None) -> str:
            nonlocal invocation_count
            with invocation_lock:
                invocation_count += 1
            handler_entered.set()
            if not release_handler.wait(timeout=5):
                raise AssertionError("timed out waiting to release active-upsert handler")
            return "active refresh protected"

        owner._run_job_type = handler  # type: ignore[method-assign]
        contender._run_job_type = handler  # type: ignore[method-assign]
        thread = Thread(target=_capture_due_run, args=(owner, owner_result))
        thread.start()
        try:
            if not handler_entered.wait(timeout=5):
                raise SystemExit("active-upsert test never entered the leased handler")
            before = {row["name"]: row for row in runtime.store.list_jobs()}["Active Refresh"]
            active_token = before["lease_token"]
            if not active_token:
                raise SystemExit(f"active-upsert fixture did not hold a lease: {dict(before)}")

            runtime.store.upsert_job("Active Refresh", 120, "refreshed_active_test", rescheduled_at)
            owner._merge_job_metadata(job_id, {"schedule_source": "concurrent_refresh"})
            refreshed = {row["name"]: row for row in runtime.store.list_jobs()}["Active Refresh"]
            if refreshed["lease_token"] != active_token or refreshed["lease_expires_at"] is None:
                raise SystemExit(f"schedule refresh cleared an active lease: {dict(refreshed)}")
            if refreshed["schedule_revision"] != before["schedule_revision"] + 1:
                raise SystemExit(f"schedule refresh did not increment its revision: {dict(refreshed)}")
            metadata = json.loads(refreshed["metadata"] or "{}")
            if metadata.get("unrelated") != "preserved" or metadata.get("schedule_source") != "concurrent_refresh":
                raise SystemExit(f"schedule refresh lost unrelated metadata: {metadata}")

            overlap = contender.run_job_now("Active Refresh")
            if "already running" not in overlap or "run skipped" not in overlap:
                raise SystemExit(f"upsert during a run permitted overlap: {overlap!r}")
            with invocation_lock:
                if invocation_count != 1:
                    raise SystemExit(f"active upsert invoked the handler more than once: {invocation_count}")
        finally:
            release_handler.set()
            thread.join(timeout=5)
        if thread.is_alive():
            raise SystemExit("active-upsert test left the owner scheduler running")
        completed = owner_result.get(timeout=5)
        if isinstance(completed, BaseException) or "Ran Active Refresh" not in str(completed):
            raise SystemExit(f"active-upsert owner did not finalize normally: {completed!r}")
        finalized = {row["name"]: row for row in runtime.store.list_jobs()}["Active Refresh"]
        if finalized["interval_minutes"] != 120 or finalized["job_type"] != "refreshed_active_test":
            raise SystemExit(f"old-run finalization overwrote the refreshed schedule: {dict(finalized)}")
        if finalized["next_run_at"] != rescheduled_at:
            raise SystemExit(f"old-run finalization overwrote the refreshed next run: {dict(finalized)}")
        if not finalized["last_run_at"]:
            raise SystemExit(f"old-run finalization did not record completion: {dict(finalized)}")
        if finalized["lease_token"] is not None or finalized["lease_expires_at"] is not None:
            raise SystemExit(f"old-run finalization did not clear its lease: {dict(finalized)}")
        if finalized["schedule_revision"] != before["schedule_revision"] + 1:
            raise SystemExit(f"old-run finalization changed the refreshed revision: {dict(finalized)}")

        subsequent = contender.run_due_jobs()
        with invocation_lock:
            final_invocation_count = invocation_count
        if subsequent != "No jobs due." or final_invocation_count != 1:
            raise SystemExit(
                f"future refreshed schedule ran again on the next tick: "
                f"output={subsequent!r}, count={final_invocation_count}"
            )


def test_concurrent_job_metadata_merges_and_history_appends() -> None:
    with TemporaryDirectory(prefix="jarvis-scheduler-metadata-merge-") as temp:
        runtime = make_temp_runtime(Path(temp))
        scheduler = Scheduler(runtime.store, runtime.vault, runtime.config)
        job_id = runtime.store.upsert_job(
            "Metadata Merge",
            60,
            "metadata_merge_test",
            iso(datetime.now() + timedelta(hours=1)),
        )
        run_at = datetime.now().replace(microsecond=0)
        runtime.store.mark_job_run(job_id, _run_marker(run_at), iso(run_at + timedelta(hours=1)))
        seeded_row = {item["name"]: item for item in runtime.store.list_jobs()}["Metadata Merge"]
        seed_time = run_at - timedelta(seconds=1)
        seed_event = {
            "occurrence_key": "legacy:" + ("a" * 64),
            "date": seed_time.date().isoformat(),
            "ran_at": _run_marker(seed_time),
            "status": "ok",
            "job_type": str(seeded_row["job_type"]),
            "schedule_identity_revision": int(seeded_row["schedule_identity_revision"]),
        }
        runtime.store.merge_job_metadata(
            job_id,
            {"unrelated": "preserved", "run_history_schema": 3},
            append={"run_history": (seed_event, 64)},
        )
        row = {item["name"]: item for item in runtime.store.list_jobs()}["Metadata Merge"]
        worker_count = 8
        start = Barrier(worker_count + 1)
        results: Queue[object] = Queue()

        def merge_config(index: int) -> None:
            try:
                start.wait(timeout=5)
                scheduler._merge_job_metadata(job_id, {f"config_{index}": index})
                results.put(True)
            except BaseException as exc:
                results.put(exc)

        def append_history(index: int) -> None:
            try:
                start.wait(timeout=5)
                results.put(scheduler._record_job_run_history(row, run_at, "ok" if index % 2 == 0 else "nothing to deliver"))
            except BaseException as exc:
                results.put(exc)

        threads = [Thread(target=merge_config, args=(index,)) for index in range(4)]
        threads.extend(Thread(target=append_history, args=(index,)) for index in range(4))
        for thread in threads:
            thread.start()
        start.wait(timeout=5)
        for thread in threads:
            thread.join(timeout=5)
        if any(thread.is_alive() for thread in threads):
            raise SystemExit("concurrent metadata merge test left a worker running")
        observed_results = [results.get(timeout=5) for _ in range(worker_count)]
        if any(result is not True for result in observed_results):
            raise SystemExit(f"concurrent metadata operation failed: {observed_results!r}")

        metadata = json.loads(
            {item["name"]: item for item in runtime.store.list_jobs()}["Metadata Merge"]["metadata"] or "{}"
        )
        if metadata.get("unrelated") != "preserved":
            raise SystemExit(f"concurrent metadata writes lost an unrelated key: {metadata}")
        if any(metadata.get(f"config_{index}") != index for index in range(4)):
            raise SystemExit(f"concurrent config merges overwrote each other: {metadata}")
        history = metadata.get("run_history")
        if not isinstance(history, list) or len(history) != 2 or history[0] != seed_event:
            raise SystemExit(f"concurrent history appends were lossy: {metadata}")
        if metadata.get("run_history_schema") != 3 or len({item["occurrence_key"] for item in history}) != 2:
            raise SystemExit(f"concurrent history appends lost typed occurrence identity: {metadata}")


def test_active_plain_occurrence_advances_across_pause_resume() -> None:
    with TemporaryDirectory(prefix="jarvis-scheduler-pause-resume-plain-") as temp:
        runtime = make_temp_runtime(Path(temp))
        base = datetime(2040, 1, 2, 3, 4, 5)
        due_at = iso(base - timedelta(minutes=5))

        for suffix, resume_before_completion in (
            ("Resume After Completion", False),
            ("Resume Before Completion", True),
        ):
            name = f"Plain {suffix}"
            job_id = runtime.store.upsert_job(name, 60, "pause_resume_plain_test", due_at)
            token = f"plain-{suffix.lower().replace(' ', '-')}"
            claimed = runtime.store.claim_job(job_id, token, 300, due_before=iso(base), now=base)
            if claimed is None:
                raise SystemExit(f"could not claim {name} fixture")
            if not runtime.store.set_job_enabled(name, False):
                raise SystemExit(f"could not pause active {name} fixture")
            if resume_before_completion and not runtime.store.set_job_enabled(name, True):
                raise SystemExit(f"could not resume active {name} fixture before completion")

            completed_at = base + timedelta(seconds=1)
            advanced_at = iso(base + timedelta(hours=1))
            if not runtime.store.mark_claimed_job_run(
                job_id,
                token,
                iso(completed_at),
                advanced_at,
                int(claimed["active_occurrence_schedule_revision"]),
                now=completed_at,
            ):
                raise SystemExit(f"active {name} fixture could not complete")
            if not resume_before_completion and not runtime.store.set_job_enabled(name, True):
                raise SystemExit(f"could not resume completed {name} fixture")

            current = next(row for row in runtime.store.list_jobs() if int(row["id"]) == job_id)
            if current["next_run_at"] != advanced_at or current["enabled"] != 1:
                raise SystemExit(f"pause/resume replayed the completed plain occurrence: {dict(current)}")
            if current["active_occurrence_key"] is not None or current["lease_token"] is not None:
                raise SystemExit(f"plain pause/resume completion left active ownership behind: {dict(current)}")

        name = "Plain Concurrent Real Reschedule"
        job_id = runtime.store.upsert_job(name, 60, "pause_resume_plain_test", due_at)
        token = "plain-real-reschedule"
        claimed = runtime.store.claim_job(job_id, token, 300, due_before=iso(base), now=base)
        if claimed is None:
            raise SystemExit("could not claim plain real-reschedule fixture")
        rescheduled_at = iso(base + timedelta(days=7))
        runtime.store.upsert_job(name, 120, "plain_rescheduled_test", rescheduled_at)
        if not runtime.store.mark_claimed_job_run(
            job_id,
            token,
            iso(base + timedelta(seconds=1)),
            iso(base + timedelta(hours=1)),
            int(claimed["active_occurrence_schedule_revision"]),
            now=base + timedelta(seconds=1),
        ):
            raise SystemExit("plain real-reschedule fixture could not complete")
        current = next(row for row in runtime.store.list_jobs() if int(row["id"]) == job_id)
        if current["next_run_at"] != rescheduled_at or current["job_type"] != "plain_rescheduled_test":
            raise SystemExit(f"plain completion overwrote a genuine concurrent reschedule: {dict(current)}")

        name = "Plain Same Due Reschedule"
        job_id = runtime.store.upsert_job(name, 60, "plain_original_test", due_at)
        token = "plain-same-due-reschedule"
        claimed = runtime.store.claim_job(job_id, token, 300, due_before=iso(base), now=base)
        if claimed is None:
            raise SystemExit("could not claim plain same-due reschedule fixture")
        runtime.store.upsert_job_with_metadata(
            name,
            180,
            "plain_same_due_rescheduled_test",
            due_at,
            {"schedule_marker": "replacement"},
        )
        if not runtime.store.mark_claimed_job_run(
            job_id,
            token,
            iso(base + timedelta(seconds=1)),
            iso(base + timedelta(hours=1)),
            int(claimed["active_occurrence_schedule_revision"]),
            now=base + timedelta(seconds=1),
        ):
            raise SystemExit("plain same-due reschedule fixture could not complete")
        current = next(row for row in runtime.store.list_jobs() if int(row["id"]) == job_id)
        metadata = json.loads(current["metadata"] or "{}")
        if (
            current["next_run_at"] != due_at
            or current["interval_minutes"] != 180
            or current["job_type"] != "plain_same_due_rescheduled_test"
            or metadata.get("schedule_marker") != "replacement"
        ):
            raise SystemExit(f"plain completion overwrote a same-due reschedule: {dict(current)}")


def test_failed_occurrence_preserves_same_due_reschedule_identity() -> None:
    with TemporaryDirectory(prefix="jarvis-scheduler-failure-same-due-") as temp:
        runtime = make_temp_runtime(Path(temp))
        base = datetime(2040, 3, 4, 5, 6, 7)
        due_at = iso(base - timedelta(minutes=5))
        name = "Failure Same Due Reschedule"
        job_id = runtime.store.upsert_job(name, 60, "failure_original_test", due_at)
        token = "failure-same-due-reschedule"
        claimed = runtime.store.claim_job(job_id, token, 300, due_before=iso(base), now=base)
        if claimed is None:
            raise SystemExit("could not claim failure same-due reschedule fixture")
        old_occurrence_key = str(claimed["active_occurrence_key"])
        runtime.store.upsert_job_with_metadata(
            name,
            240,
            "failure_same_due_rescheduled_test",
            due_at,
            {"schedule_marker": "replacement"},
        )
        if not runtime.store.mark_claimed_job_run(
            job_id,
            token,
            iso(base + timedelta(seconds=1)),
            iso(base + timedelta(minutes=5)),
            int(claimed["active_occurrence_schedule_revision"]),
            complete_occurrence=False,
            now=base + timedelta(seconds=1),
        ):
            raise SystemExit("failure same-due reschedule fixture could not release its claim")
        current = next(row for row in runtime.store.list_jobs() if int(row["id"]) == job_id)
        metadata = json.loads(current["metadata"] or "{}")
        if (
            current["next_run_at"] != due_at
            or current["interval_minutes"] != 240
            or current["job_type"] != "failure_same_due_rescheduled_test"
            or metadata.get("schedule_marker") != "replacement"
        ):
            raise SystemExit(f"failed occurrence overwrote a same-due reschedule: {dict(current)}")
        if current["lease_token"] is not None or current["active_occurrence_key"] is not None:
            raise SystemExit(f"failed stale occurrence ownership was not retired: {dict(current)}")

        replacement_token = "failure-replacement-occurrence"
        replacement = runtime.store.claim_job(
            job_id,
            replacement_token,
            300,
            due_before=iso(base),
            now=base + timedelta(seconds=2),
        )
        if replacement is None or replacement["active_occurrence_key"] == old_occurrence_key:
            raise SystemExit("same-due replacement schedule did not receive a fresh occurrence identity")
        retry_at = iso(base + timedelta(minutes=10))
        if not runtime.store.mark_claimed_job_run(
            job_id,
            replacement_token,
            iso(base + timedelta(seconds=3)),
            retry_at,
            int(replacement["active_occurrence_schedule_revision"]),
            complete_occurrence=False,
            now=base + timedelta(seconds=3),
        ):
            raise SystemExit("replacement failure could not schedule its retry")
        retried = next(row for row in runtime.store.list_jobs() if int(row["id"]) == job_id)
        if retried["next_run_at"] != retry_at or retried["active_occurrence_key"] != replacement["active_occurrence_key"]:
            raise SystemExit(f"unchanged failed occurrence lost its retry identity: {dict(retried)}")

        legacy_token = "failure-legacy-origin"
        legacy = runtime.store.claim_job(
            job_id,
            legacy_token,
            300,
            due_before=retry_at,
            now=base + timedelta(minutes=10),
        )
        if legacy is None:
            raise SystemExit("could not reclaim failure legacy-origin fixture")
        with runtime.store.connect() as conn:
            conn.execute(
                "UPDATE scheduled_jobs SET active_occurrence_schedule_identity_revision = NULL WHERE id = ?",
                (job_id,),
            )
        preserved_at = retried["next_run_at"]
        if not runtime.store.mark_claimed_job_run(
            job_id,
            legacy_token,
            iso(base + timedelta(minutes=10, seconds=1)),
            iso(base + timedelta(minutes=20)),
            int(legacy["active_occurrence_schedule_revision"]),
            now=base + timedelta(minutes=10, seconds=1),
        ):
            raise SystemExit("legacy-origin completion could not retire its lease")
        legacy_completed = next(row for row in runtime.store.list_jobs() if int(row["id"]) == job_id)
        if legacy_completed["next_run_at"] != preserved_at:
            raise SystemExit(f"missing legacy schedule identity did not fail closed: {dict(legacy_completed)}")


def test_malformed_schedule_identity_mutations_fail_closed() -> None:
    with TemporaryDirectory(prefix="jarvis-scheduler-malformed-identity-") as temp:
        runtime = make_temp_runtime(Path(temp))
        base = datetime(2040, 4, 5, 6, 7, 8)
        due_at = iso(base - timedelta(minutes=5))
        completed_at = base + timedelta(seconds=1)
        attempted_advance = iso(base + timedelta(hours=1))

        for operation in ("upsert", "metadata_upsert", "reconcile", "mark_job_run"):
            name = f"Malformed Identity {operation}"
            job_id = runtime.store.upsert_job(name, 60, f"{operation}_original", due_at)
            token = f"malformed-identity-{operation}"
            claimed = runtime.store.claim_job(job_id, token, 300, due_before=iso(base), now=base)
            if claimed is None:
                raise SystemExit(f"could not claim malformed identity {operation} fixture")
            with runtime.store.connect() as conn:
                conn.execute(
                    "UPDATE scheduled_jobs SET schedule_identity_revision = ? WHERE id = ?",
                    ("-1", job_id),
                )

            replacement_type = f"{operation}_replacement"
            if operation == "upsert":
                mutation_succeeded = runtime.store.upsert_job(name, 120, replacement_type, due_at) == job_id
            elif operation == "metadata_upsert":
                replacement = runtime.store.upsert_job_with_metadata(
                    name,
                    120,
                    replacement_type,
                    due_at,
                    {"schedule_marker": "replacement"},
                )
                mutation_succeeded = int(replacement["id"]) == job_id
            elif operation == "reconcile":
                mutation_succeeded = runtime.store.reconcile_job_schedule_if_revision(
                    job_id,
                    int(claimed["schedule_revision"]),
                    120,
                    replacement_type,
                    due_at,
                    {"schedule_marker": "replacement"},
                )
            else:
                mutation_succeeded = runtime.store.mark_job_run(
                    job_id,
                    iso(base),
                    due_at,
                )
            if not mutation_succeeded:
                raise SystemExit(f"malformed identity {operation} mutation did not commit")

            mutated = next(row for row in runtime.store.list_jobs() if int(row["id"]) == job_id)
            if type(mutated["schedule_identity_revision"]) is not int or mutated["schedule_identity_revision"] != -1:
                raise SystemExit(f"malformed identity {operation} did not use the fail-closed sentinel: {dict(mutated)}")
            if mutated["next_run_at"] != due_at or mutated["lease_token"] != token:
                raise SystemExit(f"malformed identity {operation} disturbed schedule ownership: {dict(mutated)}")
            if mutated["schedule_revision"] != claimed["schedule_revision"] + 1:
                raise SystemExit(f"malformed identity {operation} missed its operator revision: {dict(mutated)}")

            if not runtime.store.mark_claimed_job_run(
                job_id,
                token,
                iso(completed_at),
                attempted_advance,
                int(claimed["active_occurrence_schedule_revision"]),
                now=completed_at,
            ):
                raise SystemExit(f"malformed identity {operation} occurrence could not complete")
            completed = next(row for row in runtime.store.list_jobs() if int(row["id"]) == job_id)
            if completed["next_run_at"] != due_at or completed["lease_token"] is not None:
                raise SystemExit(f"old occurrence overwrote malformed identity {operation}: {dict(completed)}")

            runtime.store.upsert_job(name, 180, f"{operation}_recovered", due_at)
            recovered = next(row for row in runtime.store.list_jobs() if int(row["id"]) == job_id)
            if recovered["schedule_identity_revision"] != 0:
                raise SystemExit(f"inactive malformed identity {operation} did not recover safely: {dict(recovered)}")


def test_claim_normalizes_inactive_malformed_schedule_identity() -> None:
    with TemporaryDirectory(prefix="jarvis-scheduler-claim-identity-recovery-") as temp:
        runtime = make_temp_runtime(Path(temp))
        base = datetime(2040, 5, 6, 7, 8, 9)
        due_at = iso(base - timedelta(minutes=5))
        job_id = runtime.store.upsert_job(
            "Claim Identity Recovery",
            60,
            "claim_identity_recovery_test",
            due_at,
        )
        with runtime.store.connect() as conn:
            conn.execute(
                "UPDATE scheduled_jobs SET schedule_identity_revision = -1 WHERE id = ?",
                (job_id,),
            )

        first_token = "claim-identity-recovery-first"
        first = runtime.store.claim_job(job_id, first_token, 300, due_before=iso(base), now=base)
        if first is None:
            raise SystemExit("inactive malformed identity row could not be claimed")
        if first["schedule_identity_revision"] != 0 or first["active_occurrence_schedule_identity_revision"] != 0:
            raise SystemExit(f"claim did not safely normalize the inactive malformed identity: {dict(first)}")
        first_key = str(first["active_occurrence_key"])
        first_next_run = iso(base + timedelta(hours=1))
        if not runtime.store.mark_claimed_job_run(
            job_id,
            first_token,
            iso(base + timedelta(seconds=1)),
            first_next_run,
            int(first["active_occurrence_schedule_revision"]),
            now=base + timedelta(seconds=1),
        ):
            raise SystemExit("first normalized occurrence could not complete")
        after_first = next(row for row in runtime.store.list_jobs() if int(row["id"]) == job_id)
        if after_first["next_run_at"] != first_next_run:
            raise SystemExit(f"first normalized occurrence did not advance: {dict(after_first)}")

        second_token = "claim-identity-recovery-second"
        second_time = base + timedelta(hours=1)
        second = runtime.store.claim_job(
            job_id,
            second_token,
            300,
            due_before=first_next_run,
            now=second_time,
        )
        if second is None:
            raise SystemExit("normalized schedule could not claim its next occurrence")
        second_key = str(second["active_occurrence_key"])
        if second_key == first_key:
            raise SystemExit("normalized schedule repeated the prior occurrence key")
        second_next_run = iso(base + timedelta(hours=2))
        if not runtime.store.mark_claimed_job_run(
            job_id,
            second_token,
            iso(second_time + timedelta(seconds=1)),
            second_next_run,
            int(second["active_occurrence_schedule_revision"]),
            now=second_time + timedelta(seconds=1),
        ):
            raise SystemExit("second normalized occurrence could not complete")
        after_second = next(row for row in runtime.store.list_jobs() if int(row["id"]) == job_id)
        if after_second["next_run_at"] != second_next_run or after_second["active_occurrence_key"] is not None:
            raise SystemExit(f"normalized schedule did not advance its second occurrence: {dict(after_second)}")


def test_stale_job_owner_cannot_finalize() -> None:
    with TemporaryDirectory(prefix="jarvis-scheduler-stale-finalize-") as temp:
        runtime = make_temp_runtime(Path(temp))
        base = datetime.now().replace(microsecond=0)
        due_at = iso(base - timedelta(minutes=1))
        job_id = runtime.store.upsert_job("Stale Finalize", 60, "stale_finalize_test", due_at)
        first_token = "first-owner-token"
        second_token = "second-owner-token"
        first_claim = runtime.store.claim_job(
            job_id,
            first_token,
            1,
            due_before=iso(base),
            now=base,
        )
        if first_claim is None:
            raise SystemExit("first owner could not claim stale-finalize fixture")
        runtime.store.merge_job_metadata(job_id, {"unrelated": "preserved"})

        stale_without_reclaim = runtime.store.mark_claimed_job_run(
            job_id,
            first_token,
            iso(base + timedelta(seconds=2)),
            iso(base + timedelta(hours=1)),
            int(first_claim["schedule_revision"]),
            now=base + timedelta(seconds=2),
        )
        if stale_without_reclaim:
            raise SystemExit("an owner whose lease expired must not finalize")
        scheduler = Scheduler(runtime.store, runtime.vault, runtime.config)
        if scheduler._record_job_run_history(first_claim, base + timedelta(seconds=2), "stale output"):
            raise SystemExit("a failed finalization must not append stale run metadata")

        second_claim = runtime.store.claim_job(
            job_id,
            second_token,
            1,
            due_before=iso(base + timedelta(seconds=2)),
            now=base + timedelta(seconds=2),
        )
        if second_claim is None:
            raise SystemExit("expired claim should be recoverable by a new owner")
        stale_after_reclaim = runtime.store.mark_claimed_job_run(
            job_id,
            first_token,
            iso(base + timedelta(seconds=2)),
            iso(base + timedelta(hours=2)),
            int(first_claim["schedule_revision"]),
            now=base + timedelta(seconds=2),
        )
        if stale_after_reclaim:
            raise SystemExit("replaced stale owner must not finalize the new owner's row")
        while_second_owned = {row["name"]: row for row in runtime.store.list_jobs()}["Stale Finalize"]
        if while_second_owned["lease_token"] != second_token or while_second_owned["next_run_at"] != due_at:
            raise SystemExit(f"stale finalization mutated the replacement claim: {dict(while_second_owned)}")
        while_second_metadata = json.loads(while_second_owned["metadata"] or "{}")
        if while_second_metadata != {"unrelated": "preserved"}:
            raise SystemExit(f"stale owner mutated run metadata: {while_second_metadata}")

        second_next_run = iso(base + timedelta(hours=3))
        if not runtime.store.mark_claimed_job_run(
            job_id,
            second_token,
            iso(base + timedelta(seconds=2)),
            second_next_run,
            int(second_claim["schedule_revision"]),
            now=base + timedelta(seconds=2),
        ):
            raise SystemExit("current owner should finalize its live claim")
        finalized = {row["name"]: row for row in runtime.store.list_jobs()}["Stale Finalize"]
        if finalized["lease_token"] is not None or finalized["next_run_at"] != second_next_run:
            raise SystemExit(f"current owner finalization did not clear and reschedule atomically: {dict(finalized)}")


def test_lock_delay_uses_post_lock_time_for_lease_checks() -> None:
    with TemporaryDirectory(prefix="jarvis-scheduler-lock-delay-") as temp:
        runtime = make_temp_runtime(Path(temp))
        due_at = iso(datetime.now() - timedelta(minutes=1))
        fixtures: dict[str, tuple[int, str, int]] = {}
        for operation in ("claim", "renew", "release", "finalize"):
            job_id = runtime.store.upsert_job(
                f"Lock Delay {operation.title()}",
                60,
                f"lock_delay_{operation}_test",
                due_at,
            )
            token = f"{operation}-owner"
            claimed = runtime.store.claim_job(job_id, token, 0.25, due_before=iso(datetime.now()))
            if claimed is None:
                raise SystemExit(f"could not claim lock-delay {operation} fixture")
            fixtures[operation] = (job_id, token, int(claimed["schedule_revision"]))

        blocker = runtime.store.connect()
        blocker.execute("BEGIN IMMEDIATE")
        start = Barrier(5)
        results: Queue[tuple[str, object]] = Queue()

        def attempt(operation: str) -> None:
            job_id, token, schedule_revision = fixtures[operation]
            try:
                start.wait(timeout=5)
                if operation == "claim":
                    result = runtime.store.claim_job(job_id, "replacement-owner", 1, due_before=iso(datetime.now()))
                    results.put((operation, result))
                elif operation == "renew":
                    results.put((operation, runtime.store.renew_job_claim(job_id, token, 1)))
                elif operation == "release":
                    results.put((operation, runtime.store.release_job_claim(job_id, token)))
                else:
                    results.put(
                        (
                            operation,
                            runtime.store.mark_claimed_job_run(
                                job_id,
                                token,
                                iso(datetime.now()),
                                iso(datetime.now() + timedelta(hours=1)),
                                schedule_revision,
                            ),
                        )
                    )
            except BaseException as exc:
                results.put((operation, exc))

        threads = [Thread(target=attempt, args=(operation,)) for operation in fixtures]
        for thread in threads:
            thread.start()
        start.wait(timeout=5)
        time.sleep(0.4)
        lock_released_at = datetime.now()
        blocker.commit()
        blocker.close()
        for thread in threads:
            thread.join(timeout=5)
        if any(thread.is_alive() for thread in threads):
            raise SystemExit("post-lock lease test left a worker running")

        observed = dict(results.get(timeout=5) for _ in threads)
        if isinstance(observed.get("claim"), BaseException) or observed.get("claim") is None:
            raise SystemExit(f"claim should recover a lease that expired while waiting for the lock: {observed!r}")
        for operation in ("renew", "release", "finalize"):
            if observed.get(operation) is not False:
                raise SystemExit(f"{operation} trusted a pre-lock ownership instant: {observed!r}")
        replacement = observed["claim"]
        if datetime.fromisoformat(str(replacement["lease_expires_at"])) <= lock_released_at:
            raise SystemExit(f"replacement lease expiry was computed before lock acquisition: {dict(replacement)}")

        rows = {row["name"]: row for row in runtime.store.list_jobs()}
        if rows["Lock Delay Renew"]["lease_token"] != "renew-owner":
            raise SystemExit(f"expired renew owner mutated its row: {dict(rows['Lock Delay Renew'])}")
        if rows["Lock Delay Release"]["lease_token"] != "release-owner":
            raise SystemExit(f"expired release owner mutated its row: {dict(rows['Lock Delay Release'])}")
        if rows["Lock Delay Finalize"]["lease_token"] != "finalize-owner":
            raise SystemExit(f"expired finalize owner mutated its row: {dict(rows['Lock Delay Finalize'])}")


def test_scheduled_inbox_insert_rechecks_claim_after_lock_wait() -> None:
    with TemporaryDirectory(prefix="jarvis-scheduler-inbox-insert-lock-") as temp:
        runtime = make_temp_runtime(Path(temp))
        job_id = runtime.store.upsert_job(
            "Lock Delayed Inbox Insert",
            60,
            "inbox_ingest",
            iso(datetime.now() - timedelta(minutes=1)),
        )
        lease_token = "lock-delayed-inbox-owner"
        claimed = runtime.store.claim_job(job_id, lease_token, 0.2)
        if claimed is None:
            raise SystemExit("could not claim lock-delayed inbox fixture")
        source_key = "obsidian-inbox:" + hashlib.sha256(
            b"lock delayed scheduled inbox memory"
        ).hexdigest()
        record = MemoryRecord(
            "inbox",
            "lock delayed scheduled inbox memory",
            "lock delayed scheduled inbox memory",
            "obsidian-inbox",
            0.9,
        )

        blocker = runtime.store.connect()
        blocker.execute("BEGIN IMMEDIATE")
        attempted = Event()
        result: Queue[object] = Queue()

        def insert_as_original_owner() -> None:
            attempted.set()
            try:
                result.put(
                    runtime.store.add_memory_if_source_new_for_job_claim(
                        record,
                        source_key,
                        "obsidian-inbox",
                        job_id=job_id,
                        lease_token=lease_token,
                    )
                )
            except BaseException as exc:
                result.put(exc)

        thread = Thread(target=insert_as_original_owner)
        thread.start()
        if not attempted.wait(timeout=5):
            blocker.rollback()
            blocker.close()
            raise SystemExit("lock-delayed inbox insert did not start")
        expiry = datetime.fromisoformat(str(claimed["lease_expires_at"]))
        while datetime.now() <= expiry:
            time.sleep(0.01)
        replacement_token = "lock-delayed-inbox-replacement"
        replaced = blocker.execute(
            """
            UPDATE scheduled_jobs
            SET lease_token = ?, lease_expires_at = ?, updated_at = ?
            WHERE id = ? AND lease_token = ? AND lease_expires_at <= ?
            """,
            (
                replacement_token,
                iso(datetime.now() + timedelta(seconds=30)),
                iso(datetime.now()),
                job_id,
                lease_token,
                datetime.now().isoformat(timespec="microseconds"),
            ),
        )
        if replaced.rowcount != 1:
            blocker.rollback()
            blocker.close()
            raise SystemExit("expired inbox fixture was not legitimately replaced")
        blocker.commit()
        blocker.close()
        thread.join(timeout=5)
        if thread.is_alive():
            raise SystemExit("lock-delayed inbox insert left its stale owner running")
        stale_result = result.get(timeout=5)
        if not isinstance(stale_result, ScheduledJobLeaseAuthorityLost):
            raise SystemExit(f"stale inbox owner was not fenced after its lock wait: {stale_result!r}")

        with runtime.store.connect() as conn:
            counts = conn.execute(
                """
                SELECT
                    (SELECT COUNT(*) FROM memories WHERE body = ?) AS memories,
                    (SELECT COUNT(*) FROM ingested_sources WHERE source_key = ?) AS sources
                """,
                (record.body, source_key),
            ).fetchone()
        if counts is None or int(counts["memories"]) != 0 or int(counts["sources"]) != 0:
            raise SystemExit(f"stale inbox owner committed after replacement: {dict(counts) if counts else None}")

        inserted = runtime.store.add_memory_if_source_new_for_job_claim(
            record,
            source_key,
            "obsidian-inbox",
            job_id=job_id,
            lease_token=replacement_token,
        )
        if inserted[0] is not True or inserted[1] is None:
            raise SystemExit(f"replacement inbox owner could not commit: {inserted!r}")


def test_due_job_failure_does_not_starve_or_hot_loop() -> None:
    with TemporaryDirectory(prefix="jarvis-scheduler-row-failure-") as temp:
        runtime = make_temp_runtime(Path(temp))
        scheduler = Scheduler(runtime.store, runtime.vault, runtime.config)
        runtime.store.upsert_job(
            "First Raises",
            60,
            "first_raises_test",
            iso(datetime.now() - timedelta(minutes=2)),
        )
        runtime.store.upsert_job(
            "Second Runs",
            60,
            "second_runs_test",
            iso(datetime.now() - timedelta(minutes=1)),
        )
        invocations: list[str] = []

        def handler(job_type: str, *, row=None, now=None, lease_authority=None) -> str:
            invocations.append(job_type)
            if job_type == "first_raises_test":
                raise RuntimeError("/private/tmp/scheduler failure details must not leak")
            return "second handler completed"

        scheduler._run_job_type = handler  # type: ignore[method-assign]
        output = scheduler.run_due_jobs()
        if invocations != ["first_raises_test", "second_runs_test"]:
            raise SystemExit(f"a failed due row should not starve later jobs: {invocations}")
        for expected in ["Failed First Raises", "Job failed: RuntimeError.", "Ran Second Runs", "second handler completed"]:
            if expected not in output:
                raise SystemExit(f"isolated scheduler failure output missed {expected!r}: {output}")
        for forbidden in ["/private/tmp", "failure details", "must not leak"]:
            if forbidden in output:
                raise SystemExit(f"scheduler failure output leaked exception detail {forbidden!r}: {output}")
        assert_scheduler_recovery_guidance(output)

        rows = {row["name"]: row for row in runtime.store.list_jobs()}
        failed_row = rows["First Raises"]
        succeeded_row = rows["Second Runs"]
        failed_metadata = json.loads(failed_row["metadata"] or "{}")
        succeeded_metadata = json.loads(succeeded_row["metadata"] or "{}")
        if failed_metadata.get("last_run_status") != "failed":
            raise SystemExit(f"failed due row should record bounded failed status: {failed_metadata}")
        if succeeded_metadata.get("last_run_status") != "ok":
            raise SystemExit(f"later successful due row should record ok status: {succeeded_metadata}")
        if datetime.fromisoformat(str(failed_row["next_run_at"])) <= datetime.now():
            raise SystemExit(f"failed due row should be delayed before retry: {dict(failed_row)}")
        if failed_row["lease_token"] is not None or failed_row["lease_expires_at"] is not None:
            raise SystemExit(f"failed due row should release its lease after rescheduling: {dict(failed_row)}")

        invocations_before_retry = list(invocations)
        immediate_retry = scheduler.run_due_jobs()
        if immediate_retry != "No jobs due." or invocations != invocations_before_retry:
            raise SystemExit(
                f"failed due row must not hot-loop immediately: output={immediate_retry!r}, invocations={invocations}"
            )


def test_scheduler_claim_store_and_status_failures_name_safe_recovery() -> None:
    private_markers = (
        "/private/tmp/scheduler-guidance-secret",
        "PRIVATE_SCHEDULER_GUIDANCE_DETAIL",
    )

    with TemporaryDirectory(prefix="jarvis-scheduler-claim-guidance-") as temp:
        runtime = make_temp_runtime(Path(temp))
        scheduler = Scheduler(runtime.store, runtime.vault, runtime.config)
        runtime.store.upsert_job(
            "Claim Fails",
            60,
            "claim_fails_test",
            iso(datetime.now() - timedelta(minutes=2)),
        )
        runtime.store.upsert_job(
            "Later Runs",
            60,
            "later_runs_test",
            iso(datetime.now() - timedelta(minutes=1)),
        )
        original_claim = scheduler._claim_job
        invocations: list[str] = []

        def selective_claim(candidate, *, due_before):
            if candidate["name"] == "Claim Fails":
                raise RuntimeError(" ".join(private_markers))
            return original_claim(candidate, due_before=due_before)

        scheduler._claim_job = selective_claim  # type: ignore[method-assign]
        scheduler._run_job_type = (  # type: ignore[method-assign]
            lambda job_type, **_kwargs: invocations.append(job_type) or "later handler completed"
        )
        output = scheduler.run_due_jobs()
        if "Job claim failed: RuntimeError." not in output or "Ran Later Runs" not in output:
            raise SystemExit(f"scheduler claim failure did not preserve later due work: {output}")
        if invocations != ["later_runs_test"]:
            raise SystemExit(f"scheduler claim failure reached or starved the wrong handler: {invocations}")
        assert_scheduler_recovery_guidance(output)
        if any(marker in output for marker in private_markers):
            raise SystemExit(f"scheduler claim failure leaked private exception detail: {output}")

    with TemporaryDirectory(prefix="jarvis-scheduler-store-guidance-") as temp:
        runtime = make_temp_runtime(Path(temp))
        scheduler = Scheduler(runtime.store, runtime.vault, runtime.config)
        runtime.store.upsert_job(
            "Finalize Store Fails",
            60,
            "finalize_store_fails_test",
            iso(datetime.now() - timedelta(minutes=1)),
        )
        scheduler._run_job_type = lambda *_args, **_kwargs: "handler completed"  # type: ignore[method-assign]
        scheduler._finalize_claim_with_delivery = (  # type: ignore[method-assign]
            lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError(" ".join(private_markers)))
        )
        output = scheduler.run_due_jobs()
        if "Skipped stale claim for Finalize Store Fails." not in output:
            raise SystemExit(f"scheduler finalization store failure lost bounded status: {output}")
        assert_scheduler_recovery_guidance(output)
        if any(marker in output for marker in private_markers):
            raise SystemExit(f"scheduler finalization store failure leaked private detail: {output}")

    with TemporaryDirectory(prefix="jarvis-scheduler-status-guidance-") as temp:
        runtime = make_temp_runtime(Path(temp))
        scheduler = Scheduler(runtime.store, runtime.vault, runtime.config)
        runtime.store.upsert_job(
            "Status Recording Fails",
            60,
            "status_recording_fails_test",
            iso(datetime.now() - timedelta(minutes=1)),
        )
        scheduler._run_job_type = lambda *_args, **_kwargs: "handler completed"  # type: ignore[method-assign]
        scheduler._delivery_skip_suffix = (  # type: ignore[method-assign]
            lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError(" ".join(private_markers)))
        )
        output = scheduler.run_due_jobs()
        if "status recording failed: RuntimeError." not in output:
            raise SystemExit(f"scheduler status-recording failure lost bounded status: {output}")
        assert_scheduler_recovery_guidance(output, delivery_receipt=True)
        if any(marker in output for marker in private_markers):
            raise SystemExit(f"scheduler status-recording failure leaked private detail: {output}")


def test_scheduler_tool_results_report_actual_outcomes() -> None:
    with TemporaryDirectory(prefix="jarvis-scheduler-result-truth-") as temp:
        runtime = make_temp_runtime(Path(temp))
        original_methods = {
            name: getattr(Scheduler, name)
            for name in ("pause_job", "resume_job", "delete_job", "run_job_now")
        }
        try:
            for tool_name in ("pause_job", "resume_job", "delete_job"):
                setattr(
                    Scheduler,
                    tool_name,
                    lambda self, name: f"No job found named {name}.",
                )
                result = runtime.registry.get(tool_name).handler({"name": "Missing Job"})
                if result.ok:
                    raise SystemExit(f"{tool_name} should fail when the scheduler reports a missing job: {result}")
                if result.metadata.get("writes_files") or result.metadata.get("writes_database"):
                    raise SystemExit(f"{tool_name} missing-job metadata claimed a state change: {result.metadata}")
                setattr(Scheduler, tool_name, original_methods[tool_name])

            run_now = runtime.registry.get("run_job_now")
            run_cases = [
                ("No job found named Missing Job.", False, False),
                ("Job Busy Job is already running; run skipped.", False, False),
                ("Skipped stale claim for Stale Job; job was not started.", False, False),
                ("Ran Mystery Job:\nUnknown job type: mystery", False, True),
                ("Ran Known Job:\ncompleted", True, True),
            ]
            for output, expected_ok, expected_database_write in run_cases:
                Scheduler.run_job_now = lambda self, name, value=output: value  # type: ignore[method-assign]
                result = run_now.handler({"name": "Known Job"})
                if result.ok is not expected_ok:
                    raise SystemExit(f"run_job_now result truth mismatch for {output!r}: {result}")
                if result.metadata.get("runs_scheduled_job") is not expected_ok:
                    raise SystemExit(f"run_job_now execution metadata mismatch for {output!r}: {result.metadata}")
                if result.metadata.get("writes_database") is not expected_database_write:
                    raise SystemExit(f"run_job_now database metadata mismatch for {output!r}: {result.metadata}")
                if not expected_ok and (result.metadata.get("writes_files") or result.metadata.get("writes_notes")):
                    raise SystemExit(f"run_job_now failure metadata claimed job output writes: {result.metadata}")

            Scheduler.run_job_now = lambda self, name: f"No job found named {name}."  # type: ignore[method-assign]
            missing_run = run_now.handler({"name": "Missing Job"})
            verified, message = Verifier().verify(
                Plan("run missing job", [PlannedAction("run_job_now", {"name": "Missing Job"})]),
                [missing_run],
            )
            if verified or "run_job_now" not in message or "No job found" not in message:
                raise SystemExit(f"Verifier should reject a missing run receipt: verified={verified}, message={message!r}")
        finally:
            for name, method in original_methods.items():
                setattr(Scheduler, name, method)


def main() -> None:
    test_planner_routes_scheduler_aliases()
    test_scheduled_job_list_invalid_runtime_arguments_stop_before_scheduler_reads_or_handler()
    test_scheduler_context_reports_compaction_readiness()
    test_scheduler_reports_tolerate_malformed_job_rows()
    test_scheduler_display_redacts_hostile_truthiness_job_names()
    test_scheduler_enabled_values_fail_closed()
    test_schedule_assistant_basics_tolerates_malformed_existing_job_rows()
    test_schedule_assistant_basics_preserves_operator_state()
    test_create_job_if_missing_is_concurrent_and_case_insensitive()
    test_schedule_assistant_basics_rejects_unknown_arguments()
    test_schedule_assistant_basics_preserves_malformed_matching_row()
    test_concurrent_schedulers_claim_due_job_once()
    test_later_due_job_is_not_preclaimed_behind_slow_handler()
    test_heartbeat_keeps_long_handler_exclusively_owned()
    test_effect_boundary_atomically_renews_and_fails_closed()
    test_production_handlers_reject_lost_authority_at_first_write()
    test_inbox_mirror_boundaries_abort_without_unsafe_continuation()
    test_transient_heartbeat_error_recovers_before_production_write()
    test_sustained_heartbeat_errors_fence_production_writer_without_overlap()
    test_run_job_now_refuses_due_run_overlap()
    test_upsert_during_active_handler_preserves_lease()
    test_concurrent_job_metadata_merges_and_history_appends()
    test_active_plain_occurrence_advances_across_pause_resume()
    test_failed_occurrence_preserves_same_due_reschedule_identity()
    test_malformed_schedule_identity_mutations_fail_closed()
    test_claim_normalizes_inactive_malformed_schedule_identity()
    test_stale_job_owner_cannot_finalize()
    test_lock_delay_uses_post_lock_time_for_lease_checks()
    test_scheduled_inbox_insert_rechecks_claim_after_lock_wait()
    test_due_job_failure_does_not_starve_or_hot_loop()
    test_scheduler_claim_store_and_status_failures_name_safe_recovery()
    test_scheduler_tool_results_report_actual_outcomes()
    with TemporaryDirectory(prefix="jarvis-scheduler-basics-") as temp:
        runtime = make_temp_runtime(Path(temp))
        mission_control = Path(temp) / "Vault" / "Jarvis" / "Automations" / "Mission Control.md"
        current_context = Path(temp) / "Vault" / "Jarvis" / "Memory Tree" / "Current Context.md"
        cases = [
            "add task refresh mission control from scheduler",
            "create goal Keep scheduled state fresh because returning assistant needs orientation",
            "scheduler context refresh",
            "schedule assistant basics",
            "list scheduled jobs",
            "run job State Snapshot now",
            "scheduler context refresh",
            "run due jobs",
        ]
        for case in cases:
            result = handle_runtime_case(
                runtime,
                case,
                approved=case in {"run job State Snapshot now", "run due jobs"},
            )
            status = "ok" if result.verified else "blocked"
            print(f"[{status}] {case}")
            print(result.response[:1600])
            print()
            if case == "scheduler context refresh":
                metadata = result.tool_results[0].metadata
                assert_operator_limits(metadata, result.response, "scheduler context refresh")
                for key in SAFE_FALSE_FLAGS + ["writes_files", "writes_database", "writes_notes"]:
                    if metadata.get(key):
                        raise SystemExit(f"scheduler context refresh unsafe metadata {key}: {metadata}")
                if "Jarvis scheduler context refresh packet" not in result.response or "next safe command:" not in result.response:
                    raise SystemExit("scheduler context refresh missed packet body.")
                if metadata.get("scheduled_jobs", 0) == 0:
                    if metadata.get("refresh_state") != "REFRESH_NOT_SCHEDULED" or metadata.get("next_command") != "schedule assistant basics":
                        raise SystemExit(f"unscheduled context refresh should point to schedule assistant basics: {metadata}")
                if metadata.get("enabled_state_snapshot_jobs", 0) > 0 and metadata.get("mission_control_exists"):
                    if metadata.get("refresh_state") != "REFRESH_READY" or metadata.get("next_command") != "list scheduled jobs":
                        raise SystemExit(f"ready context refresh should point to list scheduled jobs: {metadata}")
                    if metadata.get("enabled_conversation_compaction_jobs") != 1:
                        raise SystemExit(f"ready context refresh missed enabled conversation compaction job: {metadata}")
            if case == "schedule assistant basics":
                metadata = result.tool_results[0].metadata
                if (
                    metadata.get("count") != 7
                    or metadata.get("writes_files") is not False
                    or metadata.get("writes_database") is not True
                ):
                    raise SystemExit("Schedule assistant basics missed durable scheduling metadata.")
                if (
                    metadata.get("created_jobs") != 7
                    or metadata.get("refreshed_jobs") != 0
                    or metadata.get("preserved_jobs") != 0
                    or metadata.get("idempotent_by_job_name") is not True
                ):
                    raise SystemExit(f"Schedule assistant basics missed first-run idempotency metadata: {metadata}")
                if "(created)" not in result.response or "Conversation Compaction" not in result.response:
                    raise SystemExit("Schedule assistant basics missed created-job status in output.")
                repeated_basics = runtime.registry.get("schedule_assistant_basics").handler({})
                repeated_metadata = repeated_basics.metadata
                if (
                    repeated_metadata.get("created_jobs") != 0
                    or repeated_metadata.get("refreshed_jobs") != 0
                    or repeated_metadata.get("preserved_jobs") != 7
                    or repeated_metadata.get("writes_database") is not False
                    or repeated_metadata.get("writes_files") is not False
                ):
                    raise SystemExit(
                        "Repeated schedule assistant basics should preserve existing jobs: "
                        f"{repeated_metadata}"
                    )
                jobs_after_repeat = runtime.store.list_jobs()
                if len(jobs_after_repeat) != 7:
                    raise SystemExit("Repeated schedule assistant basics should not duplicate default jobs.")
                if not any(row["name"] == "Conversation Compaction" and row["job_type"] == "conversation_compaction" for row in jobs_after_repeat):
                    raise SystemExit(f"Schedule assistant basics should include the Memory Trees compaction job: {jobs_after_repeat}")
                if "(existing; preserved)" not in repeated_basics.output:
                    raise SystemExit("Repeated schedule assistant basics missed preserved-job status in output.")
                assert_operator_limits(metadata, result.response, "schedule assistant basics")
                for key in SAFE_FALSE_FLAGS:
                    if metadata.get(key):
                        raise SystemExit(f"Schedule assistant basics unsafe metadata {key}: {metadata}")
                paused_state = runtime.registry.get("pause_job").handler({"name": "State Snapshot"})
                if not paused_state.ok or paused_state.metadata.get("writes_database") is not True:
                    raise SystemExit(f"pause State Snapshot should pause the existing scheduled job: {paused_state.metadata}")
                disabled_packet = runtime.registry.get("scheduler_context_refresh_packet").handler({})
                disabled_metadata = disabled_packet.metadata
                assert_operator_limits(disabled_metadata, disabled_packet.output, "disabled scheduler context refresh")
                if disabled_metadata.get("refresh_state") != "REFRESH_STATE_SNAPSHOT_DISABLED":
                    raise SystemExit(f"disabled state snapshot should be reported as disabled: {disabled_metadata}")
                if disabled_metadata.get("next_command") != "resume job State Snapshot":
                    raise SystemExit(f"disabled state snapshot should point to resume, not reschedule: {disabled_metadata}")
                if disabled_metadata.get("disabled_state_snapshot_jobs") != 1 or disabled_metadata.get("enabled_state_snapshot_jobs") != 0:
                    raise SystemExit(f"disabled state snapshot counters were wrong: {disabled_metadata}")
                for expected in ["disabled state snapshot jobs: 1", "resume job State Snapshot"]:
                    if expected not in disabled_packet.output:
                        raise SystemExit(f"disabled scheduler context refresh missed output: {expected}")
                resumed_state = runtime.registry.get("resume_job").handler({"name": "State Snapshot"})
                if not resumed_state.ok or resumed_state.metadata.get("writes_database") is not True:
                    raise SystemExit(f"resume State Snapshot should restore the scheduled job: {resumed_state.metadata}")
            if case == "run job State Snapshot now":
                metadata = result.tool_results[0].metadata
                if not metadata.get("writes_files") or not metadata.get("writes_database") or not metadata.get("writes_notes"):
                    raise SystemExit(f"State Snapshot job missed durable write metadata: {metadata}")
                if metadata.get("reads_private_data") is not True or metadata.get("reads_personal_data") is not True:
                    raise SystemExit(f"State Snapshot job must disclose its private state reads: {metadata}")
                assert_operator_limits(metadata, result.response, "run job State Snapshot now")
                for key in set(SAFE_FALSE_FLAGS) - {"reads_private_data", "reads_personal_data"}:
                    if metadata.get(key):
                        raise SystemExit(f"State Snapshot job unsafe metadata {key}: {metadata}")
                if "State Snapshot occurrence completed (receipt #" not in result.response:
                    raise SystemExit("State Snapshot job missed its durable occurrence receipt.")
                if "# Mission Control" in result.response or "Safe next actions:" in result.response:
                    raise SystemExit("State Snapshot job leaked projected note content into scheduler output.")
                if not current_context.exists():
                    raise SystemExit("Current Context note was not written by State Snapshot job.")
                if not mission_control.exists():
                    raise SystemExit("Mission Control note was not written by State Snapshot job.")
                mission_text = mission_control.read_text(encoding="utf-8")
                for expected in [
                    "# Mission Control",
                    "Safe next actions:",
                    "refresh mission control from scheduler",
                    "Keep scheduled state fresh",
                    "Do not auto-run shell",
                ]:
                    if expected not in mission_text:
                        raise SystemExit(f"Mission Control note missing expected scheduler context: {expected}")

        direct_daily = runtime.registry.get("schedule_daily_brief").handler({"interval_minutes": "bad"})
        if not direct_daily.ok or direct_daily.metadata.get("interval_minutes") != 1440:
            raise SystemExit("schedule_daily_brief did not sanitize a bad interval.")
        if direct_daily.metadata.get("writes_files") is not True or direct_daily.metadata.get("writes_database") is not True:
            raise SystemExit("schedule_daily_brief missed writes_files metadata.")
        assert_operator_limits(direct_daily.metadata, direct_daily.output, "schedule_daily_brief")
        for key in SAFE_FALSE_FLAGS:
            if direct_daily.metadata.get(key):
                raise SystemExit(f"schedule_daily_brief unsafe metadata {key}: {direct_daily.metadata}")

        direct_goal = runtime.registry.get("schedule_goal_nudge").handler({"interval_minutes": -100})
        if not direct_goal.ok or direct_goal.metadata.get("interval_minutes") != 1:
            raise SystemExit("schedule_goal_nudge did not clamp a low interval.")

        bool_interval = runtime.registry.get("schedule_inbox_ingest").handler({"interval_minutes": False})
        if not bool_interval.ok or bool_interval.metadata.get("interval_minutes") != 240:
            raise SystemExit("schedule_inbox_ingest should treat boolean intervals as malformed and use its default.")
        assert_operator_limits(bool_interval.metadata, bool_interval.output, "schedule_inbox_ingest boolean interval")

        direct_weekly = runtime.registry.get("schedule_weekly_review").handler({"interval_minutes": 999999999})
        if not direct_weekly.ok or direct_weekly.metadata.get("interval_minutes") != MAX_JOB_INTERVAL_MINUTES:
            raise SystemExit("schedule_weekly_review did not clamp a large interval.")

        listed = runtime.registry.get("list_scheduled_jobs").handler({})
        if not listed.ok or listed.metadata.get("writes_files") is not False:
            raise SystemExit("list_scheduled_jobs missed read-only metadata.")
        assert_operator_limits(listed.metadata, listed.output, "list_scheduled_jobs")
        for key in SAFE_FALSE_FLAGS + ["writes_files", "writes_database", "writes_notes"]:
            if listed.metadata.get(key):
                raise SystemExit(f"list_scheduled_jobs unsafe metadata {key}: {listed.metadata}")

        long_name = "Daily Brief " + ("x" * 300)
        paused = runtime.registry.get("pause_job").handler({"name": long_name})
        if len(paused.metadata.get("job_name", "")) != MAX_JOB_NAME_CHARS:
            raise SystemExit("pause_job should bound oversized job names.")
        if paused.ok or paused.metadata.get("writes_files") or paused.metadata.get("writes_database") or paused.metadata.get("writes_notes"):
            raise SystemExit(f"pause_job missing-target metadata claimed a scheduler write: {paused.metadata}")
        assert_operator_limits(paused.metadata, paused.output, "pause_job")
        for key in SAFE_FALSE_FLAGS:
            if paused.metadata.get(key):
                raise SystemExit(f"pause_job unsafe metadata {key}: {paused.metadata}")

        for tool_name in ("pause_job", "resume_job", "delete_job", "run_job_now"):
            missing_name = runtime.registry.get(tool_name).handler({"name": "   "})
            if missing_name.ok or missing_name.metadata.get("reason") != "missing_job_name":
                raise SystemExit(f"{tool_name} should reject blank job names.")
            if missing_name.metadata.get("writes_files") or missing_name.metadata.get("writes_database"):
                raise SystemExit(f"{tool_name} blank-name refusal should not write scheduler state.")
            assert_operator_limits(missing_name.metadata, missing_name.output, f"{tool_name} blank name")
            for key in SAFE_FALSE_FLAGS + ["writes_files", "writes_database", "writes_notes"]:
                if missing_name.metadata.get(key):
                    raise SystemExit(f"{tool_name} blank-name refusal unsafe metadata {key}: {missing_name.metadata}")
            assert_scheduler_refusal_handoff(
                missing_name.metadata,
                source=tool_name,
                reason="missing_job_name",
                label=f"{tool_name} blank-name refusal",
            )

            for value in [
                "/\x55sers/example/private/scheduler-job",
                "/var/folders/zc/jarvis/scheduler-job",
                "/tmp/jarvis-scheduler-job",
                HostileTruthinessText("/\x55sers/example/private/hostile-scheduler-job"),
            ]:
                path_name = runtime.registry.get(tool_name).handler({"name": value})
                if path_name.ok or path_name.metadata.get("reason") != "invalid_job_name":
                    raise SystemExit(f"{tool_name} should reject local-path-shaped job names.")
                if path_name.metadata.get("job_name") != "<local-path>":
                    raise SystemExit(f"{tool_name} should redact local-path-shaped job names in metadata: {path_name.metadata}")
                assert_no_local_path(path_name.output, f"{tool_name} path-shaped name output")
                assert_no_local_path(path_name.metadata, f"{tool_name} path-shaped name metadata")
                assert_operator_limits(path_name.metadata, path_name.output, f"{tool_name} path-shaped name")
                for key in SAFE_FALSE_FLAGS + ["writes_files", "writes_database", "writes_notes"]:
                    if path_name.metadata.get(key):
                        raise SystemExit(f"{tool_name} path-name refusal unsafe metadata {key}: {path_name.metadata}")
                assert_scheduler_refusal_handoff(
                    path_name.metadata,
                    source=tool_name,
                    reason="invalid_job_name",
                    label=f"{tool_name} path-name refusal",
                    job_name="<local-path>",
                )


def assert_operator_limits(metadata: dict, output: str, label: str) -> None:
    if metadata.get("operator_timeboxes_override_priority") is not True:
        raise SystemExit(f"{label} missed operator timebox metadata: {metadata}")
    if metadata.get("stop_times_override_priority") is not True:
        raise SystemExit(f"{label} missed stop-time metadata: {metadata}")
    if "explicit stop times" not in output or "pause commands" not in output:
        raise SystemExit(f"{label} missed scheduler operator-limit output.")


def assert_scheduler_refusal_handoff(metadata: dict, *, source: str, reason: str, label: str, job_name: str = "") -> None:
    handoff = metadata.get("scheduler_refusal_handoff")
    if not metadata.get("scheduler_refusal_handoff_ready") or not isinstance(handoff, dict):
        raise SystemExit(f"{label} missed scheduler_refusal_handoff: {metadata}")
    if handoff.get("source") != source or handoff.get("mutation") != "scheduler_job_control":
        raise SystemExit(f"{label} handoff source/mutation mismatch: {handoff}")
    if metadata.get("reason") != reason or handoff.get("reason") != reason:
        raise SystemExit(f"{label} handoff reason parity failed: {metadata}")
    if handoff.get("ready_for_operator") is not True or handoff.get("refused") is not True:
        raise SystemExit(f"{label} handoff readiness/refused mismatch: {handoff}")
    if handoff.get("changed") != []:
        raise SystemExit(f"{label} refusal should report no changed scheduler state: {handoff}")
    if job_name and (metadata.get("job_name") != job_name or handoff.get("job_name") != job_name):
        raise SystemExit(f"{label} handoff job-name parity failed: {metadata}")
    if "list scheduled jobs" not in handoff.get("next_commands", []) or not isinstance(handoff.get("retry_command"), str):
        raise SystemExit(f"{label} handoff missed recovery commands: {handoff}")
    if handoff.get("operator_timeboxes_override_priority") is not True or handoff.get("stop_times_override_priority") is not True:
        raise SystemExit(f"{label} handoff missed operator-limit precedence: {handoff}")
    boundaries = handoff.get("boundaries") or {}
    if boundaries.get("read_only") is not True:
        raise SystemExit(f"{label} handoff should stay read-only: {handoff}")
    for key in (
        "writes_files",
        "writes_database",
        "writes_memory",
        "writes_notes",
        "external_side_effect",
        "queues_approval",
        "requires_approval",
        "authorizes_execution",
        "authorizes_completion_claim",
        "approval_granted",
        "controls_computer",
        "calls_model",
        "executes_tools",
        "speaks",
        "completes_tasks",
        "runs_scheduled_job",
        "mutates_scheduled_job",
    ):
        if boundaries.get(key) is not False:
            raise SystemExit(f"{label} handoff should keep {key}=False: {handoff}")
    assert_no_local_path(handoff, f"{label} scheduler refusal handoff")


if __name__ == "__main__":
    main()
