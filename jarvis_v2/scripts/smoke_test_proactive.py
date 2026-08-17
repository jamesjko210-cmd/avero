from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from jarvis_v2.agent.planner import RuleBasedPlanner
from jarvis_v2.agent.runtime import JarvisRuntime
from jarvis_v2.automations.compaction import COMPACTION_JOB_NAME, COMPACTION_JOB_TYPE
from jarvis_v2.config import JarvisConfig
from jarvis_v2.memory.store import (
    DecisionRecord,
    GoalRecord,
    MemoryRecord,
    PersonRecord,
    PreferenceRecord,
    SkillRecord,
    TaskRecord,
)
from jarvis_v2.tools.proactive import MAX_PROACTIVE_LIMIT, MAX_STALE_DAYS, make_proactive_tools


SAFE_FALSE_FLAGS = [
    "calls_model",
    "executes_tools",
    "reads_private_data",
    "reads_personal_data",
    "writes_database",
    "writes_memory",
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


def assert_planner_routes_proactive_review_aliases() -> None:
    planner = RuleBasedPlanner()
    cases = {
        "goal nudge please": "goal_nudge",
        "show goal nudge": "goal_nudge",
        "show latest goal nudge": "goal_nudge",
        "stale goals please": "goal_nudge",
        "weekly review context please": "weekly_review_context",
        "weekly review prompt preview please": "weekly_review_prompt_preview",
        "weekly review please": "weekly_review",
        "show weekly review": "weekly_review",
    }
    for text, expected_tool in cases.items():
        actions = planner.plan(text).actions
        if [action.tool_name for action in actions] != [expected_tool]:
            raise SystemExit(f"planner missed proactive route for {text!r}: {actions!r}")


def assert_write_receipt(result, *, root: Path, label: str, prefix: str, receipt: str) -> None:
    path = result.metadata.get("path")
    path_display = result.metadata.get("path_display")
    if not isinstance(path, str) or not Path(path).exists():
        raise SystemExit(f"{label} should preserve exact saved note path metadata: {result.metadata}")
    if not isinstance(path_display, str) or not path_display.startswith(prefix):
        raise SystemExit(f"{label} should expose vault-relative path_display starting {prefix!r}: {result.metadata}")
    if f"{receipt}: {path_display}" not in result.output:
        raise SystemExit(f"{label} output should render its safe receipt path: {result.output}")
    if str(root) in result.output or "/var/folders/" in result.output or "/private/" in result.output or "/\x55sers/" in result.output:
        raise SystemExit(f"{label} output leaked a local path: {result.output}")


def assert_weekly_review_context_handoff(metadata: dict, label: str) -> None:
    handoff = metadata.get("weekly_review_context_handoff")
    if metadata.get("weekly_review_context_handoff_ready") is not True or not isinstance(handoff, dict):
        raise SystemExit(f"{label} missed weekly review context handoff: {metadata}")
    if handoff.get("source") != "weekly_review_context" or handoff.get("ready_for_operator") is not True:
        raise SystemExit(f"{label} handoff source/readiness diverged: {handoff}")
    for key in [
        "model_assist_packet_ready",
        "model_use_allowed",
        "requires_manual_review",
        "content_in_handoff",
    ]:
        if metadata.get(key) is not True or handoff.get(key) is not True:
            raise SystemExit(f"{label} missed {key}=True parity: metadata={metadata} handoff={handoff}")
    for key in ["model_use_scope", "open_tasks", "active_goals", "pending_approvals", "recent_sessions", "active_decisions", "active_preferences", "recent_tool_runs", "state_changed", "changed"]:
        if metadata.get(key) != handoff.get(key):
            raise SystemExit(f"{label} handoff {key} diverged from metadata: {metadata} vs {handoff}")
    if metadata.get("model_use_scope") != "reflection_and_planning_only":
        raise SystemExit(f"{label} should constrain model scope: {metadata}")
    if handoff.get("state_changed") is not False or handoff.get("changed") != []:
        raise SystemExit(f"{label} should report no state change: {handoff}")
    if len(handoff.get("review_questions") or []) < 4:
        raise SystemExit(f"{label} missed review questions: {handoff}")
    next_commands = handoff.get("next_commands") or {}
    for key in ["write_review", "inspect_safety", "inspect_approvals", "plan_next"]:
        if not next_commands.get(key):
            raise SystemExit(f"{label} missed next command {key}: {handoff}")
    boundaries = handoff.get("boundaries") or {}
    for key in SAFE_FALSE_FLAGS:
        if boundaries.get(key) is not False:
            raise SystemExit(f"{label} boundary should keep {key}=False: {handoff}")


def assert_weekly_review_prompt_handoff(metadata: dict, label: str) -> None:
    handoff = metadata.get("weekly_review_prompt_preview_handoff")
    if metadata.get("weekly_review_prompt_preview_handoff_ready") is not True or not isinstance(handoff, dict):
        raise SystemExit(f"{label} missed weekly review prompt handoff: {metadata}")
    if handoff.get("source") != "weekly_review_prompt_preview" or handoff.get("ready_for_operator") is not True:
        raise SystemExit(f"{label} handoff source/readiness diverged: {handoff}")
    for key in [
        "model_assist_prompt_ready",
        "model_use_allowed",
        "requires_manual_review",
        "content_in_handoff",
    ]:
        if metadata.get(key) is not True or handoff.get(key) is not True:
            raise SystemExit(f"{label} missed {key}=True parity: metadata={metadata} handoff={handoff}")
    for key in ["model_use_scope", "context_chars", "open_tasks", "active_goals", "pending_approvals", "state_changed", "changed"]:
        if metadata.get(key) != handoff.get(key):
            raise SystemExit(f"{label} handoff {key} diverged from metadata: {metadata} vs {handoff}")
    if metadata.get("model_use_scope") != "reflection_and_planning_only":
        raise SystemExit(f"{label} should constrain model scope: {metadata}")
    prompt_sections = handoff.get("prompt_sections") or {}
    for key in ["system", "user", "response_contract"]:
        if not prompt_sections.get(key):
            raise SystemExit(f"{label} missed prompt section {key}: {handoff}")
    if "Do not approve" not in prompt_sections["system"] or "Weekly review model context" not in prompt_sections["user"]:
        raise SystemExit(f"{label} prompt sections missed safety/context text: {handoff}")
    next_commands = handoff.get("next_commands") or {}
    for key in ["inspect_context", "write_review_after_manual_review", "inspect_safety", "inspect_approvals"]:
        if not next_commands.get(key):
            raise SystemExit(f"{label} missed next command {key}: {handoff}")
    boundaries = handoff.get("boundaries") or {}
    for key in SAFE_FALSE_FLAGS:
        if boundaries.get(key) is not False:
            raise SystemExit(f"{label} boundary should keep {key}=False: {handoff}")


def main() -> None:
    assert_planner_routes_proactive_review_aliases()
    with TemporaryDirectory(prefix="jarvis-proactive-") as temp:
        root = Path(temp)
        config = JarvisConfig(
            data_dir=root,
            db_path=root / "jarvis.sqlite",
            obsidian_vault=root / "Vault",
            obsidian_root="Jarvis",
            use_model_planner=False,
        )
        runtime = JarvisRuntime(config)
        (
            daily_brief,
            daily_plan,
            goal_nudge,
            weekly_review,
            weekly_review_context,
            weekly_review_prompt_preview,
            morning_startup,
            save_morning_startup,
        ) = make_proactive_tools(runtime.store, runtime.vault)
        cases = [
            "create goal Build Jarvis V2 because make a personal assistant with memory and tools by this month",
            "add step to goal 1: build proactive reviews",
            "add task review Jarvis returning-user brief priority high",
            "record decision Jarvis daily brief includes blockers because returning after hours needs orientation impact approvals and tasks stay visible",
            "add person Maya relation collaborator notes likes concise assistant updates",
            "set preference response style to direct and warm category communication",
            "run command python3 --version",
            "daily brief",
            "goal nudge",
            "weekly review context",
            "weekly review prompt preview",
            "weekly review",
            "schedule goal nudge",
            "schedule weekly review",
            "list scheduled jobs",
            "run job Weekly Review now",
        ]
        for case in cases:
            result = runtime.handle(case)
            status = "ok" if result.verified else "blocked"
            print(f"[{status}] {case}")
            print(result.response[:1200])
            print()
            if case == "daily brief":
                required = [
                    "## Open Tasks",
                    "review Jarvis returning-user brief",
                    "## Approval Queue",
                    "run command python3 --version",
                    "## Active Decisions",
                    "daily brief includes blockers",
                    "## People Context",
                    "Maya",
                    "## Active Preferences",
                    "response style",
                ]
                missing = [item for item in required if item not in result.response]
                if missing:
                    raise SystemExit(f"Daily brief missing expected context: {missing}")
            if case == "weekly review context":
                required = [
                    "Weekly review model context",
                    "Model prompt",
                    "Summarize the week into themes",
                    "Open tasks",
                    "review Jarvis returning-user brief",
                    "Goals",
                    "Build Jarvis V2",
                    "Pending approvals",
                    "run command python3 --version",
                    "Safety boundary",
                    "Review questions",
                ]
                missing = [item for item in required if item not in result.response]
                if missing:
                    raise SystemExit(f"Weekly review context missing expected context: {missing}")
                assert_weekly_review_context_handoff(result.tool_results[0].metadata, "weekly review context")
            if case == "weekly review prompt preview":
                required = [
                    "Weekly review prompt preview",
                    "System prompt",
                    "User context",
                    "Weekly review model context",
                    "Response contract",
                    "three safe next actions",
                    "No model call",
                    "Manual review is required",
                ]
                missing = [item for item in required if item not in result.response]
                if missing:
                    raise SystemExit(f"Weekly review prompt preview missing expected context: {missing}")
                assert_weekly_review_prompt_handoff(result.tool_results[0].metadata, "weekly review prompt preview")
            if case == "weekly review":
                required = [
                    "## Model Assist Packet",
                    "weekly review context",
                    "Weekly review model context",
                    "Rule: a model may suggest themes",
                ]
                missing = [item for item in required if item not in result.response]
                if missing:
                    raise SystemExit(f"Weekly review missing model-assist context: {missing}")

        reflection_dir = root / "Vault" / "Jarvis" / "Reflections"
        if not list(reflection_dir.glob("*Weekly Review.md")):
            raise SystemExit("Expected weekly review reflection was not written.")

        direct_startup = morning_startup({"limit": "bad"})
        if direct_startup.metadata.get("limit") != 6 or direct_startup.metadata.get("writes_files"):
            raise SystemExit("morning_startup should sanitize bad limits and remain read-only.")
        if "State Snapshot is not scheduled; use `schedule assistant basics`" not in direct_startup.output:
            raise SystemExit(f"morning_startup should flag missing State Snapshot before claiming background readiness: {direct_startup.output}")
        if direct_startup.metadata.get("background_ready") is not False or direct_startup.metadata.get("background_next_command") != "schedule assistant basics":
            raise SystemExit(f"morning_startup missed missing-State-Snapshot metadata: {direct_startup.metadata}")
        if direct_startup.metadata.get("state_snapshot_jobs") != 0 or direct_startup.metadata.get("enabled_state_snapshot_jobs") != 0:
            raise SystemExit(f"morning_startup State Snapshot counters should start at zero: {direct_startup.metadata}")
        for key in SAFE_FALSE_FLAGS + ["writes_notes"]:
            if direct_startup.metadata.get(key):
                raise SystemExit(f"morning_startup unsafe metadata {key}: {direct_startup.metadata}")

        bool_startup = morning_startup({"limit": False})
        if bool_startup.metadata.get("limit") != 6 or bool_startup.metadata.get("writes_files"):
            raise SystemExit("morning_startup should treat boolean limits as malformed and use its default.")

        clipped_startup = morning_startup({"limit": 999999})
        if clipped_startup.metadata.get("limit") != MAX_PROACTIVE_LIMIT:
            raise SystemExit("morning_startup should clamp huge limits.")

        runtime.store.upsert_job("State Snapshot", 360, "state_snapshot", "2099-01-01T00:00:00")
        missing_compaction_startup = morning_startup({})
        if "Conversation Compaction is not scheduled; use `schedule assistant basics`" not in missing_compaction_startup.output:
            raise SystemExit(
                "morning_startup should flag missing Conversation Compaction before claiming background readiness: "
                f"{missing_compaction_startup.output}"
            )
        if (
            missing_compaction_startup.metadata.get("background_ready") is not False
            or missing_compaction_startup.metadata.get("background_next_command") != "schedule assistant basics"
            or missing_compaction_startup.metadata.get("conversation_compaction_jobs") != 0
            or missing_compaction_startup.metadata.get("enabled_conversation_compaction_jobs") != 0
        ):
            raise SystemExit(f"morning_startup missed missing-Conversation-Compaction metadata: {missing_compaction_startup.metadata}")

        runtime.store.upsert_job(COMPACTION_JOB_NAME, 1440, COMPACTION_JOB_TYPE, "2099-01-01T00:00:00")
        if not runtime.store.set_job_enabled(COMPACTION_JOB_NAME, False):
            raise SystemExit("Conversation Compaction fixture should be pausable.")
        paused_compaction_startup = morning_startup({})
        expected_resume = f"resume job {COMPACTION_JOB_NAME}"
        if f"{COMPACTION_JOB_NAME} exists but is paused; use `{expected_resume}`" not in paused_compaction_startup.output:
            raise SystemExit(f"morning_startup should flag paused Conversation Compaction: {paused_compaction_startup.output}")
        if (
            paused_compaction_startup.metadata.get("background_ready") is not False
            or paused_compaction_startup.metadata.get("background_next_command") != expected_resume
            or paused_compaction_startup.metadata.get("conversation_compaction_jobs") != 1
            or paused_compaction_startup.metadata.get("enabled_conversation_compaction_jobs") != 0
            or paused_compaction_startup.metadata.get("disabled_conversation_compaction_jobs") != 1
        ):
            raise SystemExit(f"morning_startup missed paused-Conversation-Compaction metadata: {paused_compaction_startup.metadata}")

        if not runtime.store.set_job_enabled(COMPACTION_JOB_NAME, True):
            raise SystemExit("Conversation Compaction fixture should be resumable.")
        ready_background_startup = morning_startup({})
        if "including State Snapshot and Conversation Compaction" not in ready_background_startup.output:
            raise SystemExit(f"morning_startup should name both durable background jobs when ready: {ready_background_startup.output}")
        if ready_background_startup.metadata.get("background_ready") is not True or ready_background_startup.metadata.get("background_next_command") != "list scheduled jobs":
            raise SystemExit(f"morning_startup missed healthy background metadata: {ready_background_startup.metadata}")
        healthy_enabled_jobs = ready_background_startup.metadata.get("enabled_jobs")
        if not isinstance(healthy_enabled_jobs, int) or healthy_enabled_jobs < 2:
            raise SystemExit(f"morning_startup healthy fixture should include both background jobs: {ready_background_startup.metadata}")

        with runtime.store.connect() as conn:
            conn.execute(
                "UPDATE scheduled_jobs SET enabled = 'true' WHERE name IN (?, ?)",
                ("State Snapshot", COMPACTION_JOB_NAME),
            )
        malformed_enabled_startup = morning_startup({})
        if "State Snapshot exists but is paused; use `resume job State Snapshot`" not in malformed_enabled_startup.output:
            raise SystemExit(
                "morning_startup should fail closed on malformed enabled strings: "
                f"{malformed_enabled_startup.output}"
            )
        if (
            malformed_enabled_startup.metadata.get("background_ready") is not False
            or malformed_enabled_startup.metadata.get("background_next_command") != "resume job State Snapshot"
            or malformed_enabled_startup.metadata.get("enabled_jobs") != healthy_enabled_jobs - 2
            or malformed_enabled_startup.metadata.get("enabled_state_snapshot_jobs") != 0
            or malformed_enabled_startup.metadata.get("disabled_state_snapshot_jobs") != 1
            or malformed_enabled_startup.metadata.get("enabled_conversation_compaction_jobs") != 0
            or malformed_enabled_startup.metadata.get("disabled_conversation_compaction_jobs") != 1
        ):
            raise SystemExit(
                "morning_startup should not count malformed enabled strings as enabled: "
                f"{malformed_enabled_startup.metadata}"
            )
        if (
            "`list scheduled jobs`" in malformed_enabled_startup.output
            or "including State Snapshot and Conversation Compaction" in malformed_enabled_startup.output
        ):
            raise SystemExit(
                "morning_startup overstated malformed background rhythm as healthy: "
                f"{malformed_enabled_startup.output}"
            )
        with runtime.store.connect() as conn:
            conn.execute(
                "UPDATE scheduled_jobs SET enabled = 1 WHERE name IN (?, ?)",
                ("State Snapshot", COMPACTION_JOB_NAME),
            )

        saved_startup = save_morning_startup({"limit": -10})
        if saved_startup.metadata.get("limit") != 1 or not saved_startup.metadata.get("writes_files") or not saved_startup.metadata.get("writes_notes"):
            raise SystemExit("save_morning_startup should clamp low limits and mark writes.")
        assert_write_receipt(saved_startup, root=root, label="save_morning_startup", prefix="Daily/", receipt="Morning startup saved")
        for key in SAFE_FALSE_FLAGS:
            if saved_startup.metadata.get(key):
                raise SystemExit(f"save_morning_startup unsafe metadata {key}: {saved_startup.metadata}")

        direct_goal_nudge = goal_nudge({"stale_days": "bad"})
        if direct_goal_nudge.metadata.get("stale_days") != 7 or not direct_goal_nudge.metadata.get("writes_files") or not direct_goal_nudge.metadata.get("writes_notes"):
            raise SystemExit("goal_nudge should sanitize bad stale_days and mark writes.")

        bool_goal_nudge = goal_nudge({"stale_days": True})
        if bool_goal_nudge.metadata.get("stale_days") != 7 or not bool_goal_nudge.metadata.get("writes_files") or not bool_goal_nudge.metadata.get("writes_notes"):
            raise SystemExit("goal_nudge should treat boolean stale_days as malformed and use its default.")

        clipped_goal_nudge = goal_nudge({"stale_days": 999999})
        if clipped_goal_nudge.metadata.get("stale_days") != MAX_STALE_DAYS:
            raise SystemExit("goal_nudge should clamp huge stale_days.")

        for label, result, should_write in [
            ("daily_brief", daily_brief({}), True),
            ("daily_plan", daily_plan({}), True),
            ("goal_nudge", direct_goal_nudge, True),
            ("weekly_review", weekly_review({}), True),
            ("weekly_review_context", weekly_review_context({}), False),
            ("weekly_review_prompt_preview", weekly_review_prompt_preview({}), False),
        ]:
            if result.metadata.get("writes_files") is not should_write:
                raise SystemExit(f"{label} writes_files metadata mismatch.")
            if result.metadata.get("writes_notes") is not should_write:
                raise SystemExit(f"{label} writes_notes metadata mismatch.")
            if should_write:
                expected_prefix = "Reflections/" if label == "weekly_review" else "Daily/"
                expected_receipt = {
                    "daily_brief": "Daily brief saved",
                    "daily_plan": "Daily plan saved",
                    "goal_nudge": "Goal nudge saved",
                    "weekly_review": "Weekly review saved",
                }[label]
                assert_write_receipt(result, root=root, label=label, prefix=expected_prefix, receipt=expected_receipt)
            elif result.metadata.get("path") or result.metadata.get("path_display"):
                raise SystemExit(f"{label} should not expose write path metadata: {result.metadata}")
            for key in SAFE_FALSE_FLAGS:
                if result.metadata.get(key):
                    raise SystemExit(f"{label} unsafe metadata {key}: {result.metadata}")
            if label == "weekly_review_context":
                assert_weekly_review_context_handoff(result.metadata, label)
            if label == "weekly_review_prompt_preview":
                assert_weekly_review_prompt_handoff(result.metadata, label)

        runtime.store.add_memory(
            MemoryRecord(
                "/\x55sers/example/private/memory-category",
                "/\x55sers/example/private/memory-title",
                "legacy memory body /private/tmp/memory-body",
            )
        )
        runtime.store.add_memory(
            MemoryRecord(
                "/var/folders/zc/proactive-memory-category",
                "/tmp/proactive-memory-title",
                "legacy memory body /var/folders/zc/proactive-memory-body",
            )
        )
        runtime.store.save_skill(
            SkillRecord(
                "/\x55sers/example/private/skill-name",
                "trigger /private/tmp/skill-trigger",
                "legacy skill body",
            )
        )
        runtime.store.save_skill(
            SkillRecord(
                "/var/folders/zc/proactive-skill-name",
                "trigger /tmp/proactive-skill-trigger",
                "legacy temp-root skill body",
            )
        )
        legacy_goal_id = runtime.store.create_goal(GoalRecord("/\x55sers/example/private/goal-title", "legacy purpose"))
        runtime.store.add_goal_step(legacy_goal_id, "/private/tmp/goal-step")
        temp_goal_id = runtime.store.create_goal(GoalRecord("/var/folders/zc/proactive-goal-title", "legacy temp purpose"))
        runtime.store.add_goal_step(temp_goal_id, "/tmp/proactive-goal-step")
        runtime.store.add_task(
            TaskRecord(
                "/\x55sers/example/private/task-body",
                due="/private/tmp/task-due",
                priority="/\x55sers/example/private/task-priority",
            )
        )
        runtime.store.add_task(
            TaskRecord(
                "/var/folders/zc/proactive-task-body",
                due="/tmp/proactive-task-due",
                priority="/var/folders/zc/proactive-task-priority",
            )
        )
        runtime.store.add_pending_approval(
            "/\x55sers/example/private/session",
            "/\x55sers/example/private/approval-input",
            "/private/tmp/tool-name",
            "legacy approval reason",
        )
        runtime.store.add_pending_approval(
            "/var/folders/zc/proactive-session",
            "/tmp/proactive-approval-input",
            "/var/folders/zc/proactive-tool-name",
            "legacy temp-root approval reason",
        )
        runtime.store.add_decision(
            DecisionRecord("/\x55sers/example/private/decision-title", "/private/tmp/decision-rationale", "legacy impact")
        )
        runtime.store.add_decision(
            DecisionRecord("/var/folders/zc/proactive-decision-title", "/tmp/proactive-decision-rationale", "legacy temp impact")
        )
        runtime.store.upsert_person(PersonRecord("/\x55sers/example/private/person", "/private/tmp/relation", "legacy notes"))
        runtime.store.upsert_person(PersonRecord("/var/folders/zc/proactive-person", "/tmp/proactive-relation", "legacy temp notes"))
        runtime.store.set_preference(
            PreferenceRecord(
                "/\x55sers/example/private/preference-key",
                "/private/tmp/preference-value",
                "/\x55sers/example/private/preference-category",
            )
        )
        runtime.store.set_preference(
            PreferenceRecord(
                "/var/folders/zc/proactive-preference-key",
                "/tmp/proactive-preference-value",
                "/var/folders/zc/proactive-preference-category",
            )
        )
        runtime.store.upsert_job("/\x55sers/example/private/job-name", 60, "/private/tmp/job-type", "/\x55sers/example/private/next-run")
        runtime.store.upsert_job("/var/folders/zc/proactive-job-name", 60, "/tmp/proactive-job-type", "/var/folders/zc/proactive-next-run")
        runtime.store.log_message("/\x55sers/example/private/session", "user", "legacy message")
        runtime.store.log_message("/var/folders/zc/proactive-session", "user", "legacy temp message")
        runtime.store.log_tool_run(
            "/\x55sers/example/private/session",
            "/private/tmp/tool-run-name",
            "/\x55sers/example/private/risk",
            False,
            False,
            "legacy output",
        )
        runtime.store.log_tool_run(
            "/var/folders/zc/proactive-session",
            "/tmp/proactive-tool-run-name",
            "/var/folders/zc/proactive-risk",
            False,
            False,
            "legacy temp output",
        )

        legacy_outputs = {
            "morning_startup": morning_startup({"limit": 50}).output,
            "save_morning_startup": save_morning_startup({"limit": 50}).output,
            "daily_brief": daily_brief({}).output,
            "daily_plan": daily_plan({}).output,
            "weekly_review": weekly_review({}).output,
            "weekly_review_context": weekly_review_context({}).output,
        }
        for label, output in legacy_outputs.items():
            if "/\x55sers/" in output or "/private/" in output or "/var/folders/" in output or "/tmp/" in output:
                raise SystemExit(f"{label} leaked a raw local path:\n{output}")
            if "<local-path>" not in output:
                raise SystemExit(f"{label} did not include the expected redaction marker.")

        note_root = root / "Vault" / "Jarvis"
        note_text = "\n".join(path.read_text(encoding="utf-8") for path in note_root.rglob("*.md"))
        if "/\x55sers/" in note_text or "/private/" in note_text or "/var/folders/" in note_text or "/tmp/" in note_text:
            raise SystemExit("Saved proactive notes leaked a raw local path.")
        if "<local-path>" not in note_text:
            raise SystemExit("Saved proactive notes should include redacted legacy paths.")


if __name__ == "__main__":
    main()
