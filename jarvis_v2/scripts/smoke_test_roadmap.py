from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from jarvis_v2.scripts.test_runtime import make_temp_runtime


READ_ONLY_FLAGS = [
    "calls_model",
    "executes_tools",
    "reads_private_data",
    "reads_personal_data",
    "writes_files",
    "writes_database",
    "writes_memory",
    "writes_notes",
    "external_side_effect",
    "queues_approval",
    "requires_approval",
    "controls_computer",
    "speaks",
    "completes_tasks",
    "authorizes_execution",
    "authorizes_completion_claim",
    "approval_granted",
]


class HostileRow:
    def __init__(self, marker: str) -> None:
        self.marker = marker

    def keys(self):
        raise RuntimeError(self.marker)

    def __getitem__(self, key: str):
        raise RuntimeError(f"{self.marker}:{key}")

    def __str__(self) -> str:
        raise RuntimeError(self.marker)

    def __repr__(self) -> str:
        return f"<hostile-row {self.marker}>"


def main() -> None:
    def assert_roadmap_background(
        *,
        label: str,
        setup: callable,
        expected_text: str,
        expected_ready: bool,
        expected_command: str,
        expected_counts: dict[str, int],
    ) -> None:
        with TemporaryDirectory(prefix=f"jarvis-roadmap-{label}-") as temp:
            runtime = make_temp_runtime(Path(temp))
            setup(runtime)
            result = runtime.handle("roadmap")
            if not result.verified:
                raise SystemExit(f"Roadmap background fixture did not verify for {label}: {result.response}")
            if expected_text not in result.response:
                raise SystemExit(f"Roadmap background text missed {label}: {result.response}")
            metadata = result.tool_results[0].metadata
            if metadata.get("background_ready") is not expected_ready:
                raise SystemExit(f"Roadmap background readiness mismatch for {label}: {metadata}")
            if metadata.get("background_next_command") != expected_command:
                raise SystemExit(f"Roadmap background next command mismatch for {label}: {metadata}")
            if metadata.get("readable_scheduled_jobs", 0) + metadata.get("unreadable_scheduled_job_rows", 0) != metadata.get(
                "scheduled_jobs", metadata.get("readable_scheduled_jobs", 0)
            ):
                raise SystemExit(f"Roadmap scheduled-job readable/unreadable count mismatch for {label}: {metadata}")
            for key, expected in expected_counts.items():
                if metadata.get(key) != expected:
                    raise SystemExit(f"Roadmap background counter {key} mismatch for {label}: {metadata}")
            for key in READ_ONLY_FLAGS:
                if metadata.get(key) is not False:
                    raise SystemExit(f"Roadmap background unsafe metadata {key} for {label}: {metadata}")

    def no_setup(runtime) -> None:
        return None

    def state_snapshot_only(runtime) -> None:
        runtime.store.upsert_job("State Snapshot", 360, "state_snapshot", "2099-01-01T00:00:00")

    def paused_state_snapshot(runtime) -> None:
        runtime.store.upsert_job("State Snapshot", 360, "state_snapshot", "2099-01-01T00:00:00")
        runtime.store.upsert_job("Conversation Compaction", 1440, "conversation_compaction", "2099-01-01T00:00:00")
        paused = runtime.registry.get("pause_job").handler({"name": "State Snapshot"})
        if not paused.ok:
            raise SystemExit(f"Roadmap fixture could not pause State Snapshot: {paused.metadata}")

    def paused_conversation_compaction(runtime) -> None:
        runtime.store.upsert_job("State Snapshot", 360, "state_snapshot", "2099-01-01T00:00:00")
        runtime.store.upsert_job("Conversation Compaction", 1440, "conversation_compaction", "2099-01-01T00:00:00")
        paused = runtime.registry.get("pause_job").handler({"name": "Conversation Compaction"})
        if not paused.ok:
            raise SystemExit(f"Roadmap fixture could not pause Conversation Compaction: {paused.metadata}")

    def healthy_background(runtime) -> None:
        runtime.store.upsert_job("State Snapshot", 360, "state_snapshot", "2099-01-01T00:00:00")
        runtime.store.upsert_job("Conversation Compaction", 1440, "conversation_compaction", "2099-01-01T00:00:00")

    assert_roadmap_background(
        label="missing-state",
        setup=no_setup,
        expected_text="State Snapshot scheduled job is not configured. Next safe command: `schedule assistant basics`.",
        expected_ready=False,
        expected_command="schedule assistant basics",
        expected_counts={
            "state_snapshot_jobs": 0,
            "enabled_state_snapshot_jobs": 0,
            "conversation_compaction_jobs": 0,
            "enabled_conversation_compaction_jobs": 0,
        },
    )
    assert_roadmap_background(
        label="missing-compaction",
        setup=state_snapshot_only,
        expected_text="Conversation Compaction scheduled job is not configured. Next safe command: `schedule assistant basics`.",
        expected_ready=False,
        expected_command="schedule assistant basics",
        expected_counts={
            "state_snapshot_jobs": 1,
            "enabled_state_snapshot_jobs": 1,
            "conversation_compaction_jobs": 0,
            "enabled_conversation_compaction_jobs": 0,
        },
    )
    assert_roadmap_background(
        label="paused-state",
        setup=paused_state_snapshot,
        expected_text="State Snapshot scheduled job is paused. Next safe command: `resume job State Snapshot`.",
        expected_ready=False,
        expected_command="resume job State Snapshot",
        expected_counts={
            "state_snapshot_jobs": 1,
            "enabled_state_snapshot_jobs": 0,
            "disabled_state_snapshot_jobs": 1,
            "conversation_compaction_jobs": 1,
            "enabled_conversation_compaction_jobs": 1,
        },
    )
    assert_roadmap_background(
        label="paused-compaction",
        setup=paused_conversation_compaction,
        expected_text="Conversation Compaction scheduled job is paused. Next safe command: `resume job Conversation Compaction`.",
        expected_ready=False,
        expected_command="resume job Conversation Compaction",
        expected_counts={
            "state_snapshot_jobs": 1,
            "enabled_state_snapshot_jobs": 1,
            "conversation_compaction_jobs": 1,
            "enabled_conversation_compaction_jobs": 0,
            "disabled_conversation_compaction_jobs": 1,
        },
    )
    assert_roadmap_background(
        label="healthy-background",
        setup=healthy_background,
        expected_text="State Snapshot and Conversation Compaction are enabled.",
        expected_ready=True,
        expected_command="list scheduled jobs",
        expected_counts={
            "state_snapshot_jobs": 1,
            "enabled_state_snapshot_jobs": 1,
            "conversation_compaction_jobs": 1,
            "enabled_conversation_compaction_jobs": 1,
        },
    )

    with TemporaryDirectory(prefix="jarvis-roadmap-recent-runs-") as temp:
        runtime = make_temp_runtime(Path(temp))
        healthy_background(runtime)
        failed_run_id = runtime.store.log_tool_run(
            runtime.session_id,
            "send_kakao",
            "HIGH_RISK",
            False,
            True,
            "raw failure output should not leak from roadmap",
            metadata={"failure_kind": "transport_error", "failure_stage": "transport_timeout"},
        )
        approval_id = runtime.store.add_pending_approval(
            runtime.session_id,
            "send 가상연락처이 a telegram saying hello",
            "send_telegram",
            "explicit approval required before execution",
            {"recipient": "가상연락처이", "message": "hello"},
        )
        held_run_id = runtime.store.log_tool_run(
            runtime.session_id,
            "send_telegram",
            "HIGH_RISK",
            False,
            False,
            "raw approval held output should not leak from roadmap",
            approval_id=approval_id,
            metadata={"failure_kind": "approval-gate", "requires_confirmation": True, "toolset": "messages"},
        )

        result = runtime.handle("roadmap")
        if not result.verified:
            raise SystemExit(f"Roadmap recent-run fixture did not verify: {result.response}")
        metadata = result.tool_results[0].metadata
        for expected in [
            "recent tool runs: 2 readable / 2 total (0 ok, 1 failed/blocked, 1 approval-held, 0 unreadable hidden)",
            f"approval review: run #{held_run_id} send_telegram is approval-held",
            f"`approval readiness {approval_id}`",
            f"`approval packet {approval_id}`",
            f"`approval chain proof {approval_id}`",
            f"execution recovery: run #{failed_run_id} send_kakao needs recovery review",
            f"`execution recovery packet {failed_run_id}`",
        ]:
            if expected not in result.response:
                raise SystemExit(f"Roadmap recent-run output missing {expected}: {result.response}")
        combined = result.response + str(metadata)
        for forbidden in [
            "raw failure output should not leak from roadmap",
            "raw approval held output should not leak from roadmap",
        ]:
            if forbidden in combined:
                raise SystemExit(f"Roadmap recent-run output leaked raw audit output {forbidden}: {combined}")
        expected_counts = {
            "recent_tool_runs": 2,
            "readable_recent_tool_runs": 2,
            "unreadable_recent_tool_run_rows": 0,
            "recent_ok_tool_runs": 0,
            "recent_failed_tool_runs": 1,
            "recent_approval_held_tool_runs": 1,
            "roadmap_recent_failed_run_id": failed_run_id,
            "roadmap_recent_approval_held_run_id": held_run_id,
            "roadmap_recent_approval_held_approval_id": approval_id,
        }
        for key, expected in expected_counts.items():
            if metadata.get(key) != expected:
                raise SystemExit(f"Roadmap recent-run metadata {key} mismatch: {metadata}")
        if metadata.get("roadmap_recent_recovery_command") != f"execution recovery packet {failed_run_id}":
            raise SystemExit(f"Roadmap recent-run recovery command mismatch: {metadata}")
        if metadata.get("roadmap_recent_approval_review_commands") != [
            f"approval readiness {approval_id}",
            f"approval packet {approval_id}",
            f"approval chain proof {approval_id}",
        ]:
            raise SystemExit(f"Roadmap recent-run approval commands mismatch: {metadata}")
        for key in READ_ONLY_FLAGS:
            if metadata.get(key) is not False:
                raise SystemExit(f"Roadmap recent-run unsafe metadata {key}: {metadata}")

    with TemporaryDirectory(prefix="jarvis-roadmap-malformed-runs-") as temp:
        runtime = make_temp_runtime(Path(temp))
        leak_markers = ["ROADMAP_RUN_SECRET", "ROADMAP_RUN_OUTPUT_SECRET"]
        run_rows = [
            HostileRow(leak_markers[0]),
            {
                "id": 101,
                "tool_name": "send_telegram",
                "risk": "HIGH_RISK",
                "ok": 0,
                "approved": 0,
                "approval_id": 12,
                "output": leak_markers[1],
                "metadata": '{"failure_kind":"approval-gate","requires_confirmation":true}',
                "created_at": "2099-01-01T00:00:00Z",
            },
            {
                "id": 100,
                "tool_name": "send_kakao",
                "risk": "HIGH_RISK",
                "ok": 0,
                "approved": 1,
                "approval_id": None,
                "output": "raw failed output hidden",
                "metadata": '{"failure_kind":"transport_error","failure_stage":"transport_timeout"}',
                "created_at": "2099-01-01T00:00:00Z",
            },
        ]
        runtime.store.recent_tool_runs = lambda limit=8: run_rows[:limit]

        result = runtime.handle("roadmap")
        if not result.verified:
            raise SystemExit(f"Roadmap should tolerate malformed recent runs: {result.response}")
        metadata = result.tool_results[0].metadata
        combined = result.response + str(metadata)
        for marker in [*leak_markers, "raw failed output hidden"]:
            if marker in combined:
                raise SystemExit(f"Roadmap leaked hostile recent-run marker {marker}: {combined}")
        if "recent tool runs: 2 readable / 3 total (0 ok, 1 failed/blocked, 1 approval-held, 1 unreadable hidden)" not in result.response:
            raise SystemExit(f"Roadmap missed unreadable recent-run diagnostic: {result.response}")
        expected_counts = {
            "recent_tool_runs": 3,
            "readable_recent_tool_runs": 2,
            "unreadable_recent_tool_run_rows": 1,
            "recent_ok_tool_runs": 0,
            "recent_failed_tool_runs": 1,
            "recent_approval_held_tool_runs": 1,
            "roadmap_recent_failed_run_id": 100,
            "roadmap_recent_approval_held_run_id": 101,
            "roadmap_recent_approval_held_approval_id": 12,
        }
        for key, expected in expected_counts.items():
            if metadata.get(key) != expected:
                raise SystemExit(f"Roadmap malformed-run metadata {key} mismatch: {metadata}")
        for key in READ_ONLY_FLAGS:
            if metadata.get(key) is not False:
                raise SystemExit(f"Roadmap malformed-run unsafe metadata {key}: {metadata}")

    with TemporaryDirectory(prefix="jarvis-roadmap-malformed-rows-") as temp:
        runtime = make_temp_runtime(Path(temp))
        leak_markers = ["ROADMAP_JOB_SECRET", "ROADMAP_TASK_SECRET", "ROADMAP_GOAL_SECRET"]
        job_rows = [
            HostileRow(leak_markers[0]),
            {"id": 1, "name": "State Snapshot", "job_type": "state_snapshot", "enabled": 1},
            {"id": 2, "name": "Conversation Compaction", "job_type": "conversation_compaction", "enabled": 1},
        ]
        task_rows = [
            HostileRow(leak_markers[1]),
            {"id": 9, "body": "review roadmap row safety", "priority": "high", "status": "open"},
        ]
        goal_rows = [
            HostileRow(leak_markers[2]),
            {"id": 4, "title": "Keep roadmap safe", "status": "active"},
        ]
        runtime.store.list_jobs = lambda: job_rows
        runtime.store.list_tasks = lambda status="open", limit=5: task_rows[:limit] if status == "open" else []
        runtime.store.list_goals = lambda status="active", limit=5: goal_rows[:limit] if status == "active" else []

        result = runtime.handle("roadmap")
        if not result.verified:
            raise SystemExit(f"Roadmap should tolerate malformed local rows: {result.response}")
        metadata = result.tool_results[0].metadata
        combined = result.response + str(metadata)
        for marker in leak_markers:
            if marker in combined:
                raise SystemExit(f"Roadmap leaked hostile row marker {marker}: {combined}")
        if "Scheduled job rows include unreadable data." not in result.response:
            raise SystemExit(f"Roadmap missed unreadable scheduled-row diagnostic: {result.response}")
        if metadata.get("background_ready") is not False or metadata.get("background_next_command") != "list scheduled jobs":
            raise SystemExit(f"Roadmap should fail closed on unreadable scheduled rows: {metadata}")
        expected_counts = {
            "scheduled_jobs": 3,
            "readable_scheduled_jobs": 2,
            "unreadable_scheduled_job_rows": 1,
            "state_snapshot_jobs": 1,
            "enabled_state_snapshot_jobs": 1,
            "conversation_compaction_jobs": 1,
            "enabled_conversation_compaction_jobs": 1,
            "open_tasks": 2,
            "active_goals": 2,
        }
        for key, expected in expected_counts.items():
            if metadata.get(key) != expected:
                raise SystemExit(f"Roadmap malformed-row metadata {key} mismatch: {metadata}")
        for key in READ_ONLY_FLAGS:
            if metadata.get(key) is not False:
                raise SystemExit(f"Roadmap malformed-row unsafe metadata {key}: {metadata}")

    def assert_recommended_next_move_redacts_local_paths(*, label: str, tasks: list[object], goals: list[object]) -> None:
        with TemporaryDirectory(prefix=f"jarvis-roadmap-redacted-{label}-") as temp:
            runtime = make_temp_runtime(Path(temp))
            healthy_background(runtime)
            runtime.store.list_tasks = lambda status="open", limit=5: tasks[:limit] if status == "open" else []
            runtime.store.list_goals = lambda status="active", limit=5: goals[:limit] if status == "active" else []
            result = runtime.handle("roadmap")
            if not result.verified:
                raise SystemExit(f"Roadmap redaction fixture did not verify for {label}: {result.response}")
            combined = result.response + str(result.tool_results[0].metadata)
            if "/\x55sers/example/private" in combined:
                raise SystemExit(f"Roadmap leaked local path in recommended next move for {label}: {combined}")
            if "<local-path>" not in result.response:
                raise SystemExit(f"Roadmap did not redact local path in recommended next move for {label}: {result.response}")
            for key in READ_ONLY_FLAGS:
                if result.tool_results[0].metadata.get(key) is not False:
                    raise SystemExit(f"Roadmap redaction unsafe metadata {key} for {label}: {result.tool_results[0].metadata}")

    assert_recommended_next_move_redacts_local_paths(
        label="task",
        tasks=[
            {
                "id": 77,
                "body": "review /\x55sers/example/private/jarvis-proof.txt before expanding autonomy",
                "priority": "high",
                "status": "open",
            }
        ],
        goals=[],
    )
    assert_recommended_next_move_redacts_local_paths(
        label="goal",
        tasks=[],
        goals=[
            {
                "id": 78,
                "title": "advance /\x55sers/example/private/jarvis-goal.txt without leaking local paths",
                "status": "active",
            }
        ],
    )

    with TemporaryDirectory(prefix="jarvis-roadmap-") as temp:
        runtime = make_temp_runtime(Path(temp))
        cases = [
            "add task review Jarvis roadmap priority high",
            "create goal Build safer Jarvis autonomy because full assistant needs staged upgrades",
            "run command python3 --version",
            "roadmap",
            "roadmap report",
            "what should we build next",
        ]
        for case in cases:
            result = runtime.handle(case)
            print(f"[{'ok' if result.verified else 'blocked'}] {case}")
            print(result.response[:1800])
            print()
            if case in {"roadmap", "roadmap report", "what should we build next"}:
                for expected in [
                    "Jarvis roadmap report",
                    "agent harness",
                    "Build phases",
                    "Keep the brain inspectable",
                    "Migrate personal integrations",
                    "Improve computer-control loops",
                    "Do not skip",
                    "harness pieces",
                    "approval",
                ]:
                    if expected not in result.response:
                        raise SystemExit(f"Roadmap report missing expected text: {expected}")
                metadata = result.tool_results[0].metadata
                if metadata.get("phases") != 6 or metadata.get("tools", 0) <= 0:
                    raise SystemExit(f"Roadmap metadata missed core counts: {metadata}")
                for key in READ_ONLY_FLAGS:
                    if metadata.get(key) is not False:
                        raise SystemExit(f"Roadmap report unsafe metadata {key}: {metadata}")


if __name__ == "__main__":
    main()
