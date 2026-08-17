from __future__ import annotations

import os
import time
from pathlib import Path
from tempfile import TemporaryDirectory

from jarvis_v2.scripts.test_runtime import make_temp_runtime


def assert_contains(text: str, expected: list[str], label: str) -> None:
    missing = [item for item in expected if item not in text]
    if missing:
        raise SystemExit(f"{label} missing expected text: {missing}")


def assert_checkpoint_recovery_proof_queue(metadata: dict, label: str) -> None:
    queue = metadata.get("checkpoint_recovery_proof_queue") or []
    if not queue:
        raise SystemExit(f"{label} missed checkpoint recovery proof queue: {metadata}")
    if metadata.get("checkpoint_recovery_proof_queue_count") != len(queue):
        raise SystemExit(f"{label} checkpoint recovery proof queue count diverged: {metadata}")
    if metadata.get("checkpoint_recovery_next_proof_command") != queue[0]:
        raise SystemExit(f"{label} checkpoint recovery next proof command diverged: {metadata}")
    for expected in ["work block checkpoint", "approval readiness", "approval packet", "approval chain proof", "build delta"]:
        if not any(str(command).startswith(expected) for command in queue):
            raise SystemExit(f"{label} checkpoint recovery proof queue missed {expected}: {metadata}")


def main() -> None:
    with TemporaryDirectory(prefix="jarvis-build-progress-") as temp:
        runtime = make_temp_runtime(Path(temp))
        reflections = Path(temp) / "Vault" / "Jarvis" / "Reflections"
        setup_cases = [
            "remember that build progress should be inspectable",
            "add task review build progress report priority high",
            "create goal Keep Jarvis build visible because long autonomous sessions need progress reports",
            "add step to goal 1: summarize recent tool activity",
            "run command python3 --version",
            "get clipboard",
            "calculate 2 + 5",
        ]
        for case in setup_cases:
            result = runtime.handle(case)
            print(f"[{'ok' if result.verified else 'blocked'}] {case}")
            print(result.response[:1000])
            print()

        for case in ["build progress", "what changed in Jarvis", "what did you build"]:
            result = runtime.handle(case)
            print(f"[{'ok' if result.verified else 'blocked'}] {case}")
            print(result.response[:2400])
            print()
            if not result.verified:
                raise SystemExit(f"Expected '{case}' to run read-only.")
            assert_contains(
                result.response,
                [
                    "Jarvis build progress report",
                    "Recent execution",
                    "Safety posture",
                    "pending approvals",
                    "run_shell_command",
                    "get_clipboard",
                    "Current work state",
                    "review build progress report",
                    "Keep Jarvis build visible",
                    "Recent memory anchors",
                    "Safe next build moves",
                    "approval review",
                    "approval packet",
                    "approval chain proof",
                    "first safe handoff",
                    "approve approval",
                    "dismiss approval",
                    "focus brief",
                    "does not override the operator's explicit stop times",
                ],
                case,
            )
            metadata = result.tool_results[0].metadata
            if metadata.get("approval_handoff_first_id") != 2 or metadata.get("approval_handoff_readiness_command") != "approval readiness 2" or metadata.get("approval_handoff_proof_command") != "approval chain proof 2":
                raise SystemExit(f"{case} missed first approval handoff metadata: {metadata}")
            if metadata.get("operator_timeboxes_override_priority") is not True or metadata.get("stop_times_override_priority") is not True:
                raise SystemExit(f"{case} missed stop-time override metadata: {metadata}")

        for case in ["build delta", "what changed since checkpoint"]:
            result = runtime.handle(case)
            print(f"[{'ok' if result.verified else 'blocked'}] {case}")
            print(result.response[:2400])
            print()
            if not result.verified:
                raise SystemExit(f"Expected '{case}' to run read-only.")
            assert_contains(
                result.response,
                [
                    "Jarvis build delta",
                    "read-only",
                    "Checkpoint window",
                    "recent tool runs reviewed",
                    "Latest tool activity",
                    "run_shell_command",
                    "get_clipboard",
                    "Conversation delta",
                    "Safety delta",
                    "blocked risky runs",
                    "Safe next checks",
                    "chat continuity brief",
                    "approval review",
                    "approval packet",
                    "approval chain proof",
                    "first safe handoff",
                    "Respect the operator's explicit stop times",
                ],
                case,
            )
            metadata = result.tool_results[0].metadata
            if metadata.get("approval_handoff_first_id") != 2 or metadata.get("approval_handoff_readiness_command") != "approval readiness 2" or metadata.get("approval_handoff_proof_command") != "approval chain proof 2":
                raise SystemExit(f"Build delta missed first approval handoff metadata: {metadata}")
            if metadata.get("tool_runs", 0) < 2 or metadata.get("messages", 0) < 2:
                raise SystemExit("Build delta missed recent tool runs or messages.")
            if metadata.get("operator_timeboxes_override_priority") is not True or metadata.get("stop_times_override_priority") is not True:
                raise SystemExit(f"Build delta missed stop-time override metadata: {metadata}")
            if metadata.get("calls_model") or metadata.get("executes_tools") or metadata.get("queues_approval") or metadata.get("writes_notes") or metadata.get("controls_computer"):
                raise SystemExit("Build delta should remain read-only.")

        checkpoint = runtime.handle("work block checkpoint: continue Jarvis V2 safely")
        print(f"[{'ok' if checkpoint.verified else 'blocked'}] work block checkpoint")
        print(checkpoint.response[:2400])
        print()
        if not checkpoint.verified:
            raise SystemExit("Expected work block checkpoint to run read-only.")
        assert_contains(
            checkpoint.response,
            [
                "Jarvis work-block checkpoint",
                "resumable packet",
                "Progress evidence",
                "Verification evidence",
                "Open risks and blockers",
                "approval chain proof",
                "first safe handoff",
                "Resume packet",
                "next safe command",
                "Boundary",
                "do not continue past the operator's explicit stop time",
            ],
            "work block checkpoint",
        )
        metadata = checkpoint.tool_results[0].metadata
        if metadata.get("approval_handoff_first_id") != 2 or metadata.get("approval_handoff_readiness_command") != "approval readiness 2" or metadata.get("approval_handoff_proof_command") != "approval chain proof 2":
            raise SystemExit(f"Work-block checkpoint missed first approval handoff metadata: {metadata}")
        if metadata.get("tool_runs", 0) < 2:
            raise SystemExit("Work-block checkpoint missed recent tool runs.")
        if metadata.get("operator_timeboxes_override_priority") is not True or metadata.get("stop_times_override_priority") is not True:
            raise SystemExit(f"Work-block checkpoint missed stop-time override metadata: {metadata}")
        if metadata.get("calls_model") or metadata.get("executes_tools") or metadata.get("queues_approval") or metadata.get("writes_notes") or metadata.get("controls_computer"):
            raise SystemExit("Work-block checkpoint should remain read-only.")

        saved = runtime.handle("save build progress")
        print(f"[{'ok' if saved.verified else 'blocked'}] save build progress")
        print(saved.response[:2400])
        print()
        if not saved.verified:
            raise SystemExit("Expected save build progress to run as local-safe.")
        assert_contains(
            saved.response,
            [
                "Build progress saved",
                "Jarvis build progress report",
                "Safety posture",
                "approval chain proof",
                "first safe handoff",
                "Safe next build moves",
                "does not override the operator's explicit stop times",
            ],
            "save build progress",
        )
        saved_metadata = saved.tool_results[0].metadata
        if saved_metadata.get("operator_timeboxes_override_priority") is not True or saved_metadata.get("stop_times_override_priority") is not True:
            raise SystemExit(f"Saved build progress missed stop-time override metadata: {saved_metadata}")
        saved_files = sorted(reflections.glob("* Build Progress.md"))
        if not saved_files:
            raise SystemExit("Build progress reflection was not written.")
        note_text = saved_files[-1].read_text(encoding="utf-8")
        assert_contains(
            note_text,
            [
                "Jarvis build progress report",
                "Recent execution",
                "Safety posture",
                "approval chain proof",
                "first safe handoff",
                "Safe next build moves",
                "does not override the operator's explicit stop times",
            ],
            "saved build progress note",
        )

        saved_delta = runtime.handle("save build delta")
        print(f"[{'ok' if saved_delta.verified else 'blocked'}] save build delta")
        print(saved_delta.response[:2400])
        print()
        if not saved_delta.verified:
            raise SystemExit("Expected save build delta to run as local-safe.")
        assert_contains(
            saved_delta.response,
            [
                "Build delta saved",
                "Jarvis build delta",
                "Checkpoint window",
                "Safety delta",
                "approval chain proof",
                "first safe handoff",
                "Safe next checks",
                "Respect the operator's explicit stop times",
            ],
            "save build delta",
        )
        metadata = saved_delta.tool_results[0].metadata
        if metadata.get("tool_runs", 0) < 2 or metadata.get("messages", 0) < 2:
            raise SystemExit("Saved build delta missed recent tool runs or messages.")
        if metadata.get("operator_timeboxes_override_priority") is not True or metadata.get("stop_times_override_priority") is not True:
            raise SystemExit(f"Saved build delta missed stop-time override metadata: {metadata}")
        if (
            metadata.get("calls_model")
            or metadata.get("executes_tools")
            or metadata.get("queues_approval")
            or metadata.get("controls_computer")
            or not metadata.get("writes_notes")
        ):
            raise SystemExit("Saved build delta metadata should report only a local note write.")
        delta_files = sorted(reflections.glob("* Build Delta.md"))
        if not delta_files:
            raise SystemExit("Build delta reflection was not written.")
        delta_text = delta_files[-1].read_text(encoding="utf-8")
        assert_contains(
            delta_text,
            [
                "Jarvis build delta",
                "Checkpoint window",
                "Latest tool activity",
                "Conversation delta",
                "Safety delta",
                "approval chain proof",
                "first safe handoff",
                "Safe next checks",
                "Respect the operator's explicit stop times",
            ],
            "saved build delta note",
        )

        saved_checkpoint = runtime.handle("save work block checkpoint: continue Jarvis V2 safely")
        print(f"[{'ok' if saved_checkpoint.verified else 'blocked'}] save work block checkpoint")
        print(saved_checkpoint.response[:2400])
        print()
        if not saved_checkpoint.verified:
            raise SystemExit("Expected save work block checkpoint to run as local-safe.")
        assert_contains(
            saved_checkpoint.response,
            [
                "Work-block checkpoint saved",
                "Jarvis work-block checkpoint",
                "Verification evidence",
                "Resume packet",
                "Boundary",
                "do not continue past the operator's explicit stop time",
            ],
            "save work block checkpoint",
        )
        checkpoint_metadata = saved_checkpoint.tool_results[0].metadata
        if checkpoint_metadata.get("operator_timeboxes_override_priority") is not True or checkpoint_metadata.get("stop_times_override_priority") is not True:
            raise SystemExit(f"Saved work-block checkpoint missed stop-time override metadata: {checkpoint_metadata}")
        checkpoint_files = sorted(reflections.glob("* Work Block Checkpoint.md"))
        if not checkpoint_files:
            raise SystemExit("Work-block checkpoint reflection was not written.")
        checkpoint_text = checkpoint_files[-1].read_text(encoding="utf-8")
        assert_contains(
            checkpoint_text,
            [
                "Jarvis work-block checkpoint",
                "Progress evidence",
                "Verification evidence",
                "Open risks and blockers",
                "approval chain proof",
                "first safe handoff",
                "Resume packet",
                "Boundary",
                "do not continue past the operator's explicit stop time",
            ],
            "saved work-block checkpoint note",
        )

        recovery = runtime.handle("checkpoint recovery: continue Jarvis V2 safely")
        print(f"[{'ok' if recovery.verified else 'blocked'}] checkpoint recovery")
        print(recovery.response[:2400])
        print()
        if not recovery.verified:
            raise SystemExit("Expected checkpoint recovery to run read-only.")
        assert_contains(
            recovery.response,
            [
                "Jarvis checkpoint recovery preview",
                "Latest saved checkpoint",
                "freshness",
                "age minutes",
                "Resume-and-verify sequence",
                "Current blockers",
                "Safe next command",
                "Checkpoint recovery proof queue",
            ],
            "checkpoint recovery",
        )
        metadata = recovery.tool_results[0].metadata
        if not metadata.get("checkpoint_found") or not metadata.get("checkpoint_read"):
            raise SystemExit(f"Checkpoint recovery did not read the saved checkpoint: {metadata}")
        if metadata.get("checkpoint_freshness") not in {"fresh", "review_again", "stale", "unknown"}:
            raise SystemExit(f"Checkpoint recovery missed freshness metadata: {metadata}")
        if "checkpoint_age_minutes" not in metadata or "checkpoint_needs_review" not in metadata:
            raise SystemExit(f"Checkpoint recovery missed checkpoint age/review metadata: {metadata}")
        if metadata.get("operator_timeboxes_override_priority") is not True or metadata.get("stop_times_override_priority") is not True:
            raise SystemExit(f"Checkpoint recovery missed stop-time override metadata: {metadata}")
        assert_checkpoint_recovery_proof_queue(metadata, "Checkpoint recovery")
        if metadata.get("calls_model") or metadata.get("executes_tools") or metadata.get("queues_approval") or metadata.get("writes_notes") or metadata.get("controls_computer"):
            raise SystemExit("Checkpoint recovery should not execute or write anything.")

        recovery_cockpit = runtime.handle(
            "checkpoint recovery cockpit: continue Jarvis V2 safely stop_at=2026-06-09T08:10:00+09:00 current_time=2026-06-08T20:55:00+09:00 timezone=Asia/Seoul"
        )
        print(f"[{'ok' if recovery_cockpit.verified else 'blocked'}] checkpoint recovery cockpit")
        print(recovery_cockpit.response[:2400])
        print()
        if not recovery_cockpit.verified:
            raise SystemExit("Expected checkpoint recovery cockpit to run read-only.")
        assert_contains(
            recovery_cockpit.response,
            [
                "Jarvis checkpoint recovery cockpit",
                "Cockpit state",
                "Measured cockpit readiness scorecard",
                "required rows ready: no",
                "no_pending_approval_blockers",
                "no_failed_run_blockers",
                "Operator timebox",
                "Checkpoint recovery",
                "Recovery proof queue",
                "Required closure before normal follow-through",
                "Boundary",
            ],
            "checkpoint recovery cockpit",
        )
        metadata = recovery_cockpit.tool_results[0].metadata
        if metadata.get("cockpit_state") != "RECOVERY_COCKPIT_HELD":
            raise SystemExit(f"Checkpoint recovery cockpit should hold with pending approvals/apply gate: {metadata}")
        if metadata.get("timebox_state") != "STOP_WINDOW_ACTIVE" or metadata.get("can_continue_now") is not True:
            raise SystemExit(f"Checkpoint recovery cockpit missed active timebox metadata: {metadata}")
        if metadata.get("can_resume_local_safe_review") is not False:
            raise SystemExit(f"Checkpoint recovery cockpit should not resume with pending blockers: {metadata}")
        if not metadata.get("blockers") or metadata.get("blocker_count") != len(metadata.get("blockers")):
            raise SystemExit(f"Checkpoint recovery cockpit missed blockers metadata: {metadata}")
        scorecard_rows = metadata.get("recovery_cockpit_scorecard_rows") or []
        expected_scorecard_items = {
            "operator_timebox_active",
            "checkpoint_available",
            "checkpoint_fresh_for_review",
            "no_pending_approval_blockers",
            "no_failed_run_blockers",
            "proof_queue_ready",
            "apply_boundary_intact",
            "resume_state_consistent",
        }
        if metadata.get("recovery_cockpit_scorecard_row_count") != 8 or len(scorecard_rows) != 8:
            raise SystemExit(f"Checkpoint recovery cockpit missed scorecard rows: {metadata}")
        if {row.get("item") for row in scorecard_rows} != expected_scorecard_items:
            raise SystemExit(f"Checkpoint recovery cockpit scorecard items diverged: {metadata}")
        if metadata.get("recovery_cockpit_max_score") != 100:
            raise SystemExit(f"Checkpoint recovery cockpit missed max score: {metadata}")
        if not 0 <= int(metadata.get("recovery_cockpit_score") or -1) < 100:
            raise SystemExit(f"Checkpoint recovery cockpit held path should score below 100: {metadata}")
        if metadata.get("recovery_cockpit_required_rows_ready") is not False:
            raise SystemExit(f"Checkpoint recovery cockpit should report held scorecard rows: {metadata}")
        if any(
            row.get("ready") not in {True, False}
            or not 0 <= int(row.get("points") or 0) <= int(row.get("max_points") or 0)
            or row.get("required_before_local_safe_review") is not True
            or row.get("authorizes_risky_work") is not False
            or row.get("authorizes_execution") is not False
            for row in scorecard_rows
        ):
            raise SystemExit(f"Checkpoint recovery cockpit scorecard rows should be bounded and non-authorizing: {metadata}")
        assert_checkpoint_recovery_proof_queue(metadata, "Checkpoint recovery cockpit")
        if metadata.get("calls_model") or metadata.get("executes_tools") or metadata.get("queues_approval") or metadata.get("writes_notes") or metadata.get("controls_computer"):
            raise SystemExit("Checkpoint recovery cockpit should not execute, queue, write, or control.")

        recovery_apply = runtime.handle("checkpoint recovery apply: continue Jarvis V2 safely")
        print(f"[{'ok' if recovery_apply.verified else 'blocked'}] checkpoint recovery apply")
        print(recovery_apply.response[:2400])
        print()
        if not recovery_apply.verified:
            raise SystemExit("Expected checkpoint recovery apply packet to run read-only.")
        assert_contains(
            recovery_apply.response,
            [
                "Jarvis checkpoint recovery apply packet",
                "approval-gated planning only",
                "Approval-gated apply sequence",
                "Recovery preview excerpt",
                "Stop conditions",
                "Boundary",
            ],
            "checkpoint recovery apply",
        )
        metadata = recovery_apply.tool_results[0].metadata
        if not metadata.get("apply_requires_approval"):
            raise SystemExit(f"Checkpoint recovery apply packet should mark approval requirement: {metadata}")
        if metadata.get("checkpoint_freshness") not in {"fresh", "review_again", "stale", "unknown"}:
            raise SystemExit(f"Checkpoint recovery apply missed freshness metadata: {metadata}")
        if "checkpoint_needs_review" not in metadata:
            raise SystemExit(f"Checkpoint recovery apply missed checkpoint review metadata: {metadata}")
        if metadata.get("operator_timeboxes_override_priority") is not True or metadata.get("stop_times_override_priority") is not True:
            raise SystemExit(f"Checkpoint recovery apply missed stop-time override metadata: {metadata}")
        assert_checkpoint_recovery_proof_queue(metadata, "Checkpoint recovery apply")
        if metadata.get("calls_model") or metadata.get("executes_tools") or metadata.get("queues_approval") or metadata.get("writes_notes") or metadata.get("controls_computer"):
            raise SystemExit("Checkpoint recovery apply packet should not execute or write anything.")

        checkpoint_files = sorted(reflections.glob("* Work Block Checkpoint.md"))
        if not checkpoint_files:
            raise SystemExit("Expected saved checkpoint before continuation review-again regression check.")
        review_again_mtime = time.time() - (3 * 60 * 60)
        os.utime(checkpoint_files[-1], (review_again_mtime, review_again_mtime))
        continuation = runtime.handle("continuation packet: continue Jarvis V2 safely")
        print(f"[{'ok' if continuation.verified else 'blocked'}] continuation packet review-again checkpoint")
        print(continuation.response[:2400])
        print()
        if not continuation.verified:
            raise SystemExit("Expected continuation packet to run read-only.")
        assert_contains(
            continuation.response,
            [
                "Checkpoint recovery contract",
                "checkpoint freshness: review_again",
                "recovery required before normal follow-through: yes",
                "review-again",
            ],
            "continuation packet review-again checkpoint",
        )
        metadata = continuation.tool_results[0].metadata
        if metadata.get("checkpoint_freshness") != "review_again":
            raise SystemExit(f"Continuation packet missed review_again checkpoint freshness: {metadata}")
        if metadata.get("checkpoint_recovery_required") is not True:
            raise SystemExit(f"Continuation packet should require recovery for review_again checkpoint: {metadata}")
        if "review-again" not in str(metadata.get("checkpoint_recovery_stop_condition") or ""):
            raise SystemExit(f"Continuation packet missed review-again stop condition: {metadata}")


if __name__ == "__main__":
    main()
