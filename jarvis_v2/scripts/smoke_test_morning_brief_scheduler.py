from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from jarvis_v2.agent.planner import RuleBasedPlanner
from jarvis_v2.automations import scheduler as scheduler_module
from jarvis_v2.automations.jobs import GoalNudgeBuild
from jarvis_v2.automations.scheduler import (
    LEGACY_MORNING_BRIEF_ENV,
    MORNING_BRIEF_DEFAULT_HHMM,
    MORNING_BRIEF_ENV,
    MORNING_BRIEF_JOB_NAME,
    MORNING_BRIEF_JOB_TYPE,
    Scheduler,
    iso,
)
from jarvis_v2.scripts.test_runtime import handle_runtime_case, make_temp_runtime
from jarvis_v2.memory.store import MemoryStore


def _accepted(message_id: int = 101) -> dict:
    return {"ok": True, "result": {"message_id": message_id}}


def _goal_nudge_build(runtime, output: str) -> GoalNudgeBuild:
    snapshot = runtime.store.read_goal_nudge_snapshot()
    return GoalNudgeBuild(
        output=output,
        source_manifest_kind=snapshot.source_manifest_kind,
        source_manifest_digest=snapshot.source_manifest_digest,
    )


def _due_default_morning_brief_time(now: datetime | None = None) -> datetime:
    base = now or datetime.now()
    hour, minute = scheduler_module.parse_hhmm(MORNING_BRIEF_DEFAULT_HHMM) or (9, 0)
    due_at = base.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if due_at > base:
        due_at -= timedelta(days=1)
    return due_at


def test_parse_and_route_morning_brief_schedule() -> None:
    if scheduler_module.parse_hhmm("7:30am") != (7, 30):
        raise SystemExit("morning brief parser missed 7:30am")
    if scheduler_module.parse_hhmm("19:45") != (19, 45):
        raise SystemExit("morning brief parser missed 19:45")
    if scheduler_module.parse_hhmm("25:99") is not None:
        raise SystemExit("morning brief parser accepted invalid time")
    plan = RuleBasedPlanner().plan("schedule morning brief at 7:30am")
    if [action.tool_name for action in plan.actions] != ["schedule_morning_brief"]:
        raise SystemExit(f"planner missed morning brief schedule route: {plan}")
    if plan.actions[0].args.get("time") != "7:30am":
        raise SystemExit(f"planner did not preserve morning brief time: {plan.actions[0].args}")


def test_parse_and_route_on_demand_morning_brief_aliases() -> None:
    for command in (
        "run morning brief now",
        "send morning brief now",
        "run daily briefing now",
        "send me today's brief",
        "send me my morning brief",
        "push today's brief to my phone",
    ):
        plan = RuleBasedPlanner().plan(command)
        if [action.tool_name for action in plan.actions] != ["run_job_now"]:
            raise SystemExit(f"planner missed on-demand Morning Brief route for {command!r}: {plan}")
        if plan.actions[0].args.get("name") != MORNING_BRIEF_JOB_NAME:
            raise SystemExit(f"planner should run the scheduled Morning Brief job for {command!r}: {plan.actions[0].args}")


def test_schedule_morning_brief_tool_is_idempotent() -> None:
    with TemporaryDirectory(prefix="jarvis-morning-brief-tool-") as temp:
        runtime = make_temp_runtime(Path(temp))
        tool = runtime.registry.get("schedule_morning_brief")
        first = tool.handler({"time": "7:30am"})
        second = tool.handler({"time": "07:30"})
        if not first.ok or not second.ok:
            raise SystemExit(f"schedule_morning_brief should accept wall-clock times: {first} {second}")
        if first.metadata.get("time_hhmm") != "07:30" or second.metadata.get("time_hhmm") != "07:30":
            raise SystemExit(f"schedule_morning_brief should normalize time metadata: {first.metadata} {second.metadata}")
        jobs = runtime.store.list_jobs()
        morning_jobs = [job for job in jobs if job["name"] == MORNING_BRIEF_JOB_NAME]
        if len(morning_jobs) != 1 or morning_jobs[0]["job_type"] != MORNING_BRIEF_JOB_TYPE:
            raise SystemExit(f"schedule_morning_brief should upsert one Morning Brief job: {jobs}")
        committed_metadata = json.loads(morning_jobs[0]["metadata"] or "{}")
        committed_hhmm = datetime.fromisoformat(str(morning_jobs[0]["next_run_at"])).strftime("%H:%M")
        if (
            committed_metadata.get("configured_time_hhmm") != committed_hhmm
            or second.metadata.get("time_hhmm") != committed_hhmm
        ):
            raise SystemExit(
                f"schedule receipt, committed schedule, and metadata diverged: {second.metadata} {morning_jobs[0]}"
            )
        bad = tool.handler({"time": "25:99"})
        if bad.ok or bad.metadata.get("reason") != "invalid_time":
            raise SystemExit(f"schedule_morning_brief should reject invalid times locally: {bad.metadata}")


def test_concurrent_morning_brief_schedule_and_metadata_are_atomic() -> None:
    with TemporaryDirectory(prefix="jarvis-morning-brief-concurrent-") as temp:
        runtime = make_temp_runtime(Path(temp))
        now = datetime(2026, 7, 11, 6, 0)

        def schedule(hhmm: str) -> tuple[int, str] | None:
            store = MemoryStore(runtime.config.db_path)
            return Scheduler(store, runtime.vault, runtime.config).schedule_morning_brief(hhmm, now=now)

        requested = ["07:30" if index % 2 == 0 else "09:00" for index in range(20)]
        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(schedule, requested))
        if any(result is None for result in results):
            raise SystemExit(f"concurrent Morning Brief schedule unexpectedly failed: {results}")

        rows = [row for row in runtime.store.list_jobs() if row["name"] == MORNING_BRIEF_JOB_NAME]
        if len(rows) != 1:
            raise SystemExit(f"concurrent Morning Brief scheduling created duplicate rows: {rows}")
        row = rows[0]
        metadata = json.loads(row["metadata"] or "{}")
        scheduled_hhmm = datetime.fromisoformat(str(row["next_run_at"])).strftime("%H:%M")
        if metadata.get("configured_time_hhmm") != scheduled_hhmm:
            raise SystemExit(f"concurrent Morning Brief schedule split from metadata: {row}")
        if len({int(result[0]) for result in results if result is not None}) != 1:
            raise SystemExit(f"concurrent Morning Brief callers observed different job ids: {results}")


def test_morning_brief_schedule_metadata_failure_rolls_back() -> None:
    with TemporaryDirectory(prefix="jarvis-morning-brief-rollback-") as temp:
        runtime = make_temp_runtime(Path(temp))
        try:
            runtime.store.upsert_job_with_metadata(
                MORNING_BRIEF_JOB_NAME,
                scheduler_module.MORNING_BRIEF_INTERVAL_MINUTES,
                MORNING_BRIEF_JOB_TYPE,
                "2026-07-12T09:00:00",
                {"configured_time_hhmm": object()},
            )
        except TypeError:
            pass
        else:
            raise SystemExit("unserializable Morning Brief metadata did not fail")
        if runtime.store.list_jobs():
            raise SystemExit("failed Morning Brief metadata serialization left a partial new job")

        scheduled = Scheduler(runtime.store, runtime.vault, runtime.config).schedule_morning_brief(
            "07:30", now=datetime(2026, 7, 11, 6, 0)
        )
        if scheduled is None:
            raise SystemExit("Morning Brief rollback setup did not schedule")
        before = dict(runtime.store.list_jobs()[0])
        try:
            runtime.store.upsert_job_with_metadata(
                MORNING_BRIEF_JOB_NAME,
                scheduler_module.MORNING_BRIEF_INTERVAL_MINUTES,
                MORNING_BRIEF_JOB_TYPE,
                "2026-07-12T09:00:00",
                {"configured_time_hhmm": object()},
            )
        except TypeError:
            pass
        else:
            raise SystemExit("unserializable Morning Brief refresh metadata did not fail")
        after = dict(runtime.store.list_jobs()[0])
        if after["next_run_at"] != before["next_run_at"] or after["metadata"] != before["metadata"]:
            raise SystemExit(f"failed Morning Brief refresh partially committed: {before} -> {after}")


def test_env_refresh_revalidates_schedule_before_metadata_merge() -> None:
    with TemporaryDirectory(prefix="jarvis-morning-brief-env-race-") as temp:
        runtime = make_temp_runtime(Path(temp))
        scheduler = Scheduler(runtime.store, runtime.vault, runtime.config)
        now = datetime(2026, 7, 11, 6, 0)
        if scheduler.schedule_morning_brief("09:00", now=now) is None:
            raise SystemExit("Morning Brief env race setup did not schedule")

        original_merge = runtime.store.merge_job_metadata
        raced = False

        def racing_merge(job_id, updates=None, **kwargs):
            nonlocal raced
            if not raced and kwargs.get("expected_next_run_at") is not None:
                raced = True
                other = MemoryStore(runtime.config.db_path)
                other.upsert_job_with_metadata(
                    MORNING_BRIEF_JOB_NAME,
                    scheduler_module.MORNING_BRIEF_INTERVAL_MINUTES,
                    MORNING_BRIEF_JOB_TYPE,
                    "2026-07-12T07:30:00",
                    {
                        "configured_time_hhmm": "07:30",
                        "schedule_source": "concurrent-test",
                        "once_per_day_guard": True,
                    },
                )
            return original_merge(job_id, updates, **kwargs)

        runtime.store.merge_job_metadata = racing_merge  # type: ignore[method-assign]
        try:
            with mock.patch.dict(
                os.environ,
                {MORNING_BRIEF_ENV: "09:00", LEGACY_MORNING_BRIEF_ENV: ""},
                clear=False,
            ):
                scheduler.ensure_env_morning_brief(now=now)
        finally:
            runtime.store.merge_job_metadata = original_merge  # type: ignore[method-assign]
        if not raced:
            raise SystemExit("Morning Brief env refresh did not exercise the schedule race")
        row = [row for row in runtime.store.list_jobs() if row["name"] == MORNING_BRIEF_JOB_NAME][0]
        metadata = json.loads(row["metadata"] or "{}")
        scheduled_hhmm = datetime.fromisoformat(str(row["next_run_at"])).strftime("%H:%M")
        if scheduled_hhmm != "09:00" or metadata.get("configured_time_hhmm") != scheduled_hhmm:
            raise SystemExit(f"Morning Brief env race left split schedule metadata: {row}")


def test_due_morning_brief_sends_once_to_owner_channel() -> None:
    with TemporaryDirectory(prefix="jarvis-morning-brief-due-") as temp:
        runtime = make_temp_runtime(Path(temp))
        scheduler = Scheduler(runtime.store, runtime.vault, runtime.config)
        due_at = _due_default_morning_brief_time()
        runtime.store.upsert_job(
            MORNING_BRIEF_JOB_NAME,
            scheduler_module.MORNING_BRIEF_INTERVAL_MINUTES,
            MORNING_BRIEF_JOB_TYPE,
            iso(due_at),
        )
        sent: list[str] = []
        with mock.patch.object(scheduler_module, "_build_live_daily_brief", return_value="Good morning from Jarvis."), \
             mock.patch.object(scheduler_module, "_send_owner_telegram", side_effect=lambda text: sent.append(text) or _accepted()):
            first = scheduler.run_due_jobs()
            second = scheduler.run_due_jobs()
        if sent != ["Good morning from Jarvis."]:
            raise SystemExit(f"due Morning Brief should send exactly once: {sent}")
        if "Morning Brief accepted by Telegram API" not in first:
            raise SystemExit(f"due Morning Brief output wrong: {first}")
        if "Morning Brief sent to Telegram" in first:
            raise SystemExit(f"due Morning Brief output should not overclaim phone delivery: {first}")
        if second != "No jobs due.":
            raise SystemExit(f"Morning Brief should be idempotent across immediate second tick: {second}")
        next_run = runtime.store.list_jobs()[0]["next_run_at"]
        if datetime.fromisoformat(str(next_run)) <= datetime.now():
            raise SystemExit(f"Morning Brief next run should be moved to the future: {next_run}")
        metadata = json.loads(runtime.store.list_jobs()[0]["metadata"] or "{}")
        if metadata.get("last_sent_date") != datetime.now().date().isoformat():
            raise SystemExit(f"Morning Brief should persist last-sent date after success: {metadata}")

        runtime.store.upsert_job(
            MORNING_BRIEF_JOB_NAME,
            scheduler_module.MORNING_BRIEF_INTERVAL_MINUTES,
            MORNING_BRIEF_JOB_TYPE,
            iso(_due_default_morning_brief_time()),
        )
        restarted_scheduler = Scheduler(runtime.store, runtime.vault, runtime.config)
        with mock.patch.object(scheduler_module, "_build_live_daily_brief", return_value="Duplicate should not compose."), \
             mock.patch.object(scheduler_module, "_send_owner_telegram", side_effect=lambda text: sent.append(text) or _accepted()):
            duplicate = restarted_scheduler.run_due_jobs()
        if sent != ["Good morning from Jarvis."]:
            raise SystemExit(f"Morning Brief once-per-day guard should survive restart: {sent}")
        if "already accepted by Telegram API today" not in duplicate:
            raise SystemExit(f"Morning Brief duplicate due run should explain skipped duplicate: {duplicate}")
        metadata = json.loads(runtime.store.list_jobs()[0]["metadata"] or "{}")
        history = metadata.get("run_history")
        statuses = [event.get("status") for event in history] if isinstance(history, list) else []
        if statuses != ["skipped"] or not history[0].get("occurrence_key", "").startswith("morning:"):
            raise SystemExit(f"Morning Brief run history should converge one duplicate-skip receipt: {metadata}")
        if "Good morning" in json.dumps(history):
            raise SystemExit(f"Morning Brief run history must not store brief content: {metadata}")


def test_morning_brief_delivery_failures_are_actionable_and_scrubbed() -> None:
    def run_due_with_send(send_side_effect):
        with TemporaryDirectory(prefix="jarvis-morning-brief-delivery-fail-") as temp:
            runtime = make_temp_runtime(Path(temp))
            scheduler = Scheduler(runtime.store, runtime.vault, runtime.config)
            runtime.store.upsert_job(
                MORNING_BRIEF_JOB_NAME,
                scheduler_module.MORNING_BRIEF_INTERVAL_MINUTES,
                MORNING_BRIEF_JOB_TYPE,
                iso(_due_default_morning_brief_time()),
            )
            with mock.patch.object(scheduler_module, "_build_live_daily_brief", return_value="Good morning."), \
                 mock.patch.object(scheduler_module, "_send_owner_telegram", side_effect=send_side_effect):
                output = scheduler.run_due_jobs()
            metadata = json.loads(runtime.store.list_jobs()[0]["metadata"] or "{}")
            return output, metadata

    missing_owner, missing_metadata = run_due_with_send(
        lambda text: {"ok": False, "error": "missing_or_invalid_owner"}
    )
    for fragment in [
        "Morning Brief not sent:",
        "owner Telegram chat is not configured",
        "JARVIS_OWNER_TELEGRAM",
        "setup check",
        "channel status",
    ]:
        if fragment not in missing_owner:
            raise SystemExit(f"missing-owner Morning Brief failure missed {fragment!r}: {missing_owner}")
    if missing_metadata.get("last_run_status") != "failed":
        raise SystemExit(f"missing-owner Morning Brief failure should record failed status: {missing_metadata}")

    raw_error, _ = run_due_with_send(
        lambda text: {"ok": False, "error": "/\x55sers/example/private SHOULD NOT APPEAR"}
    )
    for fragment in ["Morning Brief not sent:", "Telegram delivery failed", "setup check", "channel status"]:
        if fragment not in raw_error:
            raise SystemExit(f"raw-error Morning Brief failure missed {fragment!r}: {raw_error}")
    for forbidden in ["/Users", "SHOULD NOT APPEAR", "operator/private"]:
        if forbidden in raw_error:
            raise SystemExit(f"raw-error Morning Brief failure leaked diagnostic {forbidden!r}: {raw_error}")

    raised_error, raised_metadata = run_due_with_send(
        lambda text: (_ for _ in ()).throw(RuntimeError("/private/tmp SHOULD NOT APPEAR"))
    )
    for fragment in ["Morning Brief Telegram outcome is unknown", "may have been accepted", "will not automatically resend"]:
        if fragment not in raised_error:
            raise SystemExit(f"raised Morning Brief failure missed {fragment!r}: {raised_error}")
    if raised_metadata.get("last_run_status") != "failed":
        raise SystemExit(f"unknown Morning Brief outcome should record failed run status: {raised_metadata}")
    for forbidden in ["/private", "SHOULD NOT APPEAR", "RuntimeError"]:
        if forbidden in raised_error:
            raise SystemExit(f"raised Morning Brief failure leaked diagnostic {forbidden!r}: {raised_error}")


def test_missing_owner_preflight_does_not_fence_same_day_morning_brief_retry() -> None:
    with TemporaryDirectory(prefix="jarvis-morning-brief-owner-preflight-") as temp:
        runtime = make_temp_runtime(Path(temp))
        scheduler = Scheduler(runtime.store, runtime.vault, runtime.config)
        runtime.store.upsert_job(
            MORNING_BRIEF_JOB_NAME,
            scheduler_module.MORNING_BRIEF_INTERVAL_MINUTES,
            MORNING_BRIEF_JOB_TYPE,
            iso(_due_default_morning_brief_time()),
        )
        composed: list[str] = []
        sent: list[str] = []
        with mock.patch(
            "jarvis_v2.automations.telegram_control._owner_chat_id",
            return_value="",
        ), mock.patch.object(
            scheduler_module,
            "_build_live_daily_brief",
            side_effect=lambda _config: composed.append("built") or "Brief content.",
        ), mock.patch.object(
            scheduler_module,
            "_send_owner_telegram",
            side_effect=lambda text: sent.append(text) or _accepted(),
        ):
            missing_owner = scheduler.run_due_jobs()

        for expected in ("Morning Brief not sent:", "JARVIS_OWNER_TELEGRAM", "setup check", "channel status"):
            if expected not in missing_owner:
                raise SystemExit(f"missing-owner preflight missed recovery guidance {expected!r}: {missing_owner}")
        if composed or sent:
            raise SystemExit(f"missing owner must not compose or send a Morning Brief: composed={composed} sent={sent}")
        if any(runtime.store.scheduled_delivery_state_counts().values()):
            raise SystemExit("missing owner must not create a terminal Morning Brief delivery receipt")

        with mock.patch(
            "jarvis_v2.automations.telegram_control._owner_chat_id",
            return_value="owner-smoke",
        ), mock.patch.object(
            scheduler_module,
            "_build_live_daily_brief",
            side_effect=lambda _config: composed.append("built") or "Brief content.",
        ), mock.patch.object(
            scheduler_module,
            "_send_owner_telegram",
            side_effect=lambda text: sent.append(text) or _accepted(),
        ):
            retry = scheduler.run_job_now(MORNING_BRIEF_JOB_NAME)

        if "Morning Brief accepted by Telegram API" not in retry:
            raise SystemExit(f"same-day owner-config recovery should deliver the Morning Brief: {retry}")
        if composed != ["built"] or sent != ["Brief content."]:
            raise SystemExit(f"same-day retry should compose/send exactly once: composed={composed} sent={sent}")


def test_legacy_same_day_guard_prevents_upgrade_duplicate() -> None:
    with TemporaryDirectory(prefix="jarvis-morning-brief-legacy-outbox-bridge-") as temp:
        runtime = make_temp_runtime(Path(temp))
        scheduler = Scheduler(runtime.store, runtime.vault, runtime.config)
        job_id = runtime.store.upsert_job(
            MORNING_BRIEF_JOB_NAME,
            scheduler_module.MORNING_BRIEF_INTERVAL_MINUTES,
            MORNING_BRIEF_JOB_TYPE,
            iso(_due_default_morning_brief_time()),
        )
        today = datetime.now().date().isoformat()
        runtime.store.merge_job_metadata(job_id, {"last_sent_date": today, "last_send_status": "ok"})
        with mock.patch.object(scheduler_module, "_build_live_daily_brief") as build, \
             mock.patch.object(scheduler_module, "_send_owner_telegram") as send:
            output = scheduler.run_due_jobs()
        if build.called or send.called:
            raise SystemExit("legacy same-day Morning Brief guard must not compose or send during outbox migration")
        if "legacy guard" not in output or "skipped duplicate" not in output:
            raise SystemExit(f"legacy same-day Morning Brief guard should explain migration skip: {output}")
        if runtime.store.scheduled_delivery_state_counts().get("accepted") != 0:
            raise SystemExit("legacy guard must not invent a new Telegram API receipt")
        metadata = json.loads(runtime.store.list_jobs()[0]["metadata"] or "{}")
        history = metadata.get("run_history") or []
        if not history or history[-1].get("status") != "skipped":
            raise SystemExit(f"legacy guard should record a skipped scheduler run: {metadata}")


def test_on_demand_morning_brief_runtime_sends_once_to_owner_channel() -> None:
    with TemporaryDirectory(prefix="jarvis-morning-brief-now-") as temp:
        runtime = make_temp_runtime(Path(temp))
        scheduled = runtime.registry.get("schedule_morning_brief").handler({"time": "7:30am"})
        if not scheduled.ok:
            raise SystemExit(f"schedule_morning_brief should create the job before on-demand run: {scheduled}")

        built: list[object] = []
        sent: list[str] = []
        with mock.patch.object(
            scheduler_module,
            "_build_live_daily_brief",
            side_effect=lambda config: built.append(config) or "Good morning from on-demand Jarvis.",
        ), mock.patch.object(
            scheduler_module,
            "_send_owner_telegram",
            side_effect=lambda text: sent.append(text) or _accepted(),
        ):
            result = handle_runtime_case(runtime, "run morning brief now", approved=True)

        if not result.verified:
            raise SystemExit(f"on-demand Morning Brief should verify: {result.response}")
        if [tool_result.tool_name for tool_result in result.tool_results] != ["run_job_now"]:
            raise SystemExit(f"on-demand Morning Brief should reuse run_job_now: {result.tool_results}")
        tool_result = result.tool_results[0]
        if not tool_result.ok or "Morning Brief accepted by Telegram API" not in tool_result.output:
            raise SystemExit(f"on-demand Morning Brief should send through scheduler job: {tool_result}")
        if "Morning Brief sent to Telegram" in tool_result.output:
            raise SystemExit(f"on-demand Morning Brief should not overclaim phone delivery: {tool_result}")
        if len(built) != 1:
            raise SystemExit(f"on-demand Morning Brief should compose exactly once: {built}")
        if sent != ["Good morning from on-demand Jarvis."]:
            raise SystemExit(f"on-demand Morning Brief should send exactly once to owner channel: {sent}")
        if tool_result.metadata.get("job_name") != MORNING_BRIEF_JOB_NAME:
            raise SystemExit(f"run_job_now metadata should preserve Morning Brief job name: {tool_result.metadata}")
        if not tool_result.metadata.get("writes_database") or not tool_result.metadata.get("writes_notes"):
            raise SystemExit(f"run_job_now should keep scheduled-job run metadata: {tool_result.metadata}")
        for key in ("reads_private_data", "reads_personal_data", "external_side_effect", "may_send_owner_telegram", "owner_telegram_only"):
            if tool_result.metadata.get(key) is not True:
                raise SystemExit(f"on-demand Morning Brief should disclose {key}: {tool_result.metadata}")
        if tool_result.metadata.get("requires_approval") or tool_result.metadata.get("queues_approval"):
            raise SystemExit(f"trusted owner Morning Brief path should remain approval-free: {tool_result.metadata}")


def test_on_demand_phone_brief_alias_sends_once_to_owner_channel() -> None:
    with TemporaryDirectory(prefix="jarvis-morning-brief-phone-alias-") as temp:
        runtime = make_temp_runtime(Path(temp))
        scheduled = runtime.registry.get("schedule_morning_brief").handler({"time": "7:30am"})
        if not scheduled.ok:
            raise SystemExit(f"schedule_morning_brief should create the job before phone alias run: {scheduled}")

        built: list[object] = []
        sent: list[str] = []
        with mock.patch.object(
            scheduler_module,
            "_build_live_daily_brief",
            side_effect=lambda config: built.append(config) or "Good morning from the phone alias.",
        ), mock.patch.object(
            scheduler_module,
            "_send_owner_telegram",
            side_effect=lambda text: sent.append(text) or _accepted(),
        ):
            result = handle_runtime_case(runtime, "push today's brief to my phone", approved=True)

        if not result.verified:
            raise SystemExit(f"phone brief alias should verify: {result.response}")
        if [tool_result.tool_name for tool_result in result.tool_results] != ["run_job_now"]:
            raise SystemExit(f"phone brief alias should reuse run_job_now: {result.tool_results}")
        if len(built) != 1 or sent != ["Good morning from the phone alias."]:
            raise SystemExit(f"phone brief alias should compose and send exactly once: built={built} sent={sent}")


def test_not_due_and_env_opt_in_do_not_send() -> None:
    with TemporaryDirectory(prefix="jarvis-morning-brief-env-") as temp:
        runtime = make_temp_runtime(Path(temp))
        scheduler = Scheduler(runtime.store, runtime.vault, runtime.config)
        sent: list[str] = []
        with mock.patch.dict(os.environ, {MORNING_BRIEF_ENV: "06:45"}, clear=False), \
             mock.patch.object(scheduler_module, "_build_live_daily_brief", return_value="Should not send yet."), \
             mock.patch.object(scheduler_module, "_send_owner_telegram", side_effect=lambda text: sent.append(text) or _accepted()):
            output = scheduler.run_due_jobs()
        jobs = runtime.store.list_jobs()
        if output != "No jobs due." or sent:
            raise SystemExit(f"env-created Morning Brief should not send before due: {output} {sent}")
        if len(jobs) != 1 or jobs[0]["name"] != MORNING_BRIEF_JOB_NAME or jobs[0]["job_type"] != MORNING_BRIEF_JOB_TYPE:
            raise SystemExit(f"env opt-in should create one Morning Brief job: {jobs}")


def test_env_default_legacy_and_empty_disable() -> None:
    with TemporaryDirectory(prefix="jarvis-morning-brief-env-default-") as temp:
        runtime = make_temp_runtime(Path(temp))
        scheduler = Scheduler(runtime.store, runtime.vault, runtime.config)
        frozen = datetime(2026, 7, 3, 8, 0)

        with mock.patch.dict(os.environ, {}, clear=True):
            scheduler.ensure_env_morning_brief(now=frozen)
        jobs = runtime.store.list_jobs()
        if len(jobs) != 1 or jobs[0]["name"] != MORNING_BRIEF_JOB_NAME:
            raise SystemExit(f"unset env should create the default Morning Brief job: {jobs}")
        if datetime.fromisoformat(str(jobs[0]["next_run_at"])).strftime("%H:%M") != MORNING_BRIEF_DEFAULT_HHMM:
            raise SystemExit(f"unset env should default Morning Brief to {MORNING_BRIEF_DEFAULT_HHMM}: {jobs[0]['next_run_at']}")
        metadata = json.loads(jobs[0]["metadata"] or "{}")
        if metadata.get("configured_time_hhmm") != MORNING_BRIEF_DEFAULT_HHMM or metadata.get("schedule_source") != "default":
            raise SystemExit(f"default Morning Brief metadata wrong: {metadata}")

        with mock.patch.dict(os.environ, {MORNING_BRIEF_ENV: ""}, clear=True):
            scheduler.ensure_env_morning_brief(now=frozen)
        disabled = runtime.store.list_jobs()[0]
        if int(disabled["enabled"]) != 0:
            raise SystemExit(f"empty {MORNING_BRIEF_ENV} should disable Morning Brief: {dict(disabled)}")
        disabled_metadata = json.loads(disabled["metadata"] or "{}")
        if disabled_metadata.get("disabled_by_env") is not True:
            raise SystemExit(f"empty {MORNING_BRIEF_ENV} should record disable provenance: {disabled_metadata}")

        with mock.patch.dict(os.environ, {MORNING_BRIEF_ENV: "10:15"}, clear=True):
            scheduler.ensure_env_morning_brief(now=frozen)
        restored = runtime.store.list_jobs()[0]
        restored_metadata = json.loads(restored["metadata"] or "{}")
        if int(restored["enabled"]) != 1 or restored_metadata.get("disabled_by_env") is not False:
            raise SystemExit(f"restored {MORNING_BRIEF_ENV} should re-enable its own disabled job: {dict(restored)}")
        if datetime.fromisoformat(str(restored["next_run_at"])).strftime("%H:%M") != "10:15":
            raise SystemExit(f"restored {MORNING_BRIEF_ENV} should apply its new cadence: {dict(restored)}")

    with TemporaryDirectory(prefix="jarvis-morning-brief-env-legacy-") as temp:
        runtime = make_temp_runtime(Path(temp))
        scheduler = Scheduler(runtime.store, runtime.vault, runtime.config)
        with mock.patch.dict(os.environ, {LEGACY_MORNING_BRIEF_ENV: "06:45"}, clear=True):
            scheduler.ensure_env_morning_brief(now=datetime(2026, 7, 3, 6, 0))
        jobs = runtime.store.list_jobs()
        if len(jobs) != 1 or datetime.fromisoformat(str(jobs[0]["next_run_at"])).strftime("%H:%M") != "06:45":
            raise SystemExit(f"legacy env should still schedule Morning Brief at 06:45: {jobs}")
        metadata = json.loads(jobs[0]["metadata"] or "{}")
        if metadata.get("schedule_source") != LEGACY_MORNING_BRIEF_ENV:
            raise SystemExit(f"legacy env metadata should record source: {metadata}")


def test_env_morning_brief_normalizes_malformed_enabled_row() -> None:
    with TemporaryDirectory(prefix="jarvis-morning-brief-malformed-enabled-") as temp:
        runtime = make_temp_runtime(Path(temp))
        scheduler = Scheduler(runtime.store, runtime.vault, runtime.config)
        frozen = datetime(2026, 7, 3, 8, 0)
        runtime.store.upsert_job(
            MORNING_BRIEF_JOB_NAME,
            scheduler_module.MORNING_BRIEF_INTERVAL_MINUTES,
            MORNING_BRIEF_JOB_TYPE,
            iso(scheduler_module.next_daily_run(9, 0, now=frozen)),
        )
        with runtime.store.connect() as conn:
            conn.execute(
                "UPDATE scheduled_jobs SET enabled = 'true' WHERE name = ?",
                (MORNING_BRIEF_JOB_NAME,),
            )
        with mock.patch.dict(os.environ, {MORNING_BRIEF_ENV: "09:00"}, clear=True):
            scheduler.ensure_env_morning_brief(now=frozen)
        jobs = runtime.store.list_jobs()
        if len(jobs) != 1 or int(jobs[0]["enabled"]) != 1:
            raise SystemExit(f"env Morning Brief should normalize malformed enabled row back to 1: {jobs}")
        metadata = json.loads(jobs[0]["metadata"] or "{}")
        if metadata.get("configured_time_hhmm") != "09:00" or metadata.get("schedule_source") != MORNING_BRIEF_ENV:
            raise SystemExit(f"normalized Morning Brief row should refresh env metadata: {metadata}")


def test_env_reconciliation_preserves_explicit_morning_brief_pause() -> None:
    with TemporaryDirectory(prefix="jarvis-morning-brief-pause-intent-") as temp:
        runtime = make_temp_runtime(Path(temp))
        scheduler = Scheduler(runtime.store, runtime.vault, runtime.config)
        frozen = datetime(2026, 7, 3, 8, 0)
        with mock.patch.dict(os.environ, {MORNING_BRIEF_ENV: "09:00"}, clear=True):
            scheduler.ensure_env_morning_brief(now=frozen)
        if scheduler.pause_job(MORNING_BRIEF_JOB_NAME) != f"Paused {MORNING_BRIEF_JOB_NAME}.":
            raise SystemExit("Morning Brief pause fixture did not pause the job")
        paused_baseline = dict(runtime.store.list_jobs()[0])

        sent: list[str] = []
        with mock.patch.dict(os.environ, {MORNING_BRIEF_ENV: ""}, clear=True):
            scheduler.ensure_env_morning_brief(now=frozen)
        blank_tick = dict(runtime.store.list_jobs()[0])
        blank_metadata = json.loads(blank_tick["metadata"] or "{}")
        if blank_metadata.get("disabled_by_env") is True:
            raise SystemExit("empty environment relabeled the operator's explicit Morning Brief pause")
        with mock.patch.dict(os.environ, {MORNING_BRIEF_ENV: "10:15"}, clear=True), \
             mock.patch.object(scheduler_module, "_build_live_daily_brief", return_value="Should stay paused."), \
             mock.patch.object(scheduler_module, "_send_owner_telegram", side_effect=lambda text: sent.append(text) or _accepted()):
            output = scheduler.run_due_jobs()
        paused = dict(runtime.store.list_jobs()[0])
        if output != "No jobs due." or sent:
            raise SystemExit(f"paused Morning Brief ran during environment reconciliation: {output} {sent}")
        if int(paused["enabled"]) != 0:
            raise SystemExit(f"environment reconciliation re-enabled Morning Brief: {paused}")
        for key in ("next_run_at", "schedule_revision", "metadata"):
            if paused[key] != paused_baseline[key]:
                raise SystemExit(
                    f"paused Morning Brief mutated {key}: {paused_baseline} -> {paused}"
                )

        if scheduler.resume_job(MORNING_BRIEF_JOB_NAME) != f"Resumed {MORNING_BRIEF_JOB_NAME}.":
            raise SystemExit("Morning Brief resume fixture did not resume the job")
        with mock.patch.dict(os.environ, {MORNING_BRIEF_ENV: "10:15"}, clear=True):
            scheduler.ensure_env_morning_brief(now=frozen)
        resumed = dict(runtime.store.list_jobs()[0])
        resumed_metadata = json.loads(resumed["metadata"] or "{}")
        if int(resumed["enabled"]) != 1 or resumed_metadata.get("configured_time_hhmm") != "10:15":
            raise SystemExit(f"explicitly resumed Morning Brief did not reconcile configuration: {resumed}")
        if datetime.fromisoformat(str(resumed["next_run_at"])).strftime("%H:%M") != "10:15":
            raise SystemExit(f"resumed Morning Brief kept stale cadence: {resumed}")

        scheduler.pause_job(MORNING_BRIEF_JOB_NAME)
        with runtime.store.connect() as conn:
            metadata = json.loads(conn.execute(
                "SELECT metadata FROM scheduled_jobs WHERE name = ?",
                (MORNING_BRIEF_JOB_NAME,),
            ).fetchone()["metadata"] or "{}")
            metadata["disabled_by_env"] = "false"
            conn.execute(
                "UPDATE scheduled_jobs SET metadata = ? WHERE name = ?",
                (json.dumps(metadata, sort_keys=True), MORNING_BRIEF_JOB_NAME),
            )
        with mock.patch.dict(os.environ, {MORNING_BRIEF_ENV: "11:30"}, clear=True):
            scheduler.ensure_env_morning_brief(now=frozen)
        malformed = dict(runtime.store.list_jobs()[0])
        if int(malformed["enabled"]) != 0:
            raise SystemExit(f"malformed disable provenance re-enabled Morning Brief: {malformed}")


def test_env_reconciliation_cannot_overwrite_concurrent_pause() -> None:
    with TemporaryDirectory(prefix="jarvis-morning-brief-pause-race-") as temp:
        runtime = make_temp_runtime(Path(temp))
        scheduler = Scheduler(runtime.store, runtime.vault, runtime.config)
        frozen = datetime(2026, 7, 3, 8, 0)
        with mock.patch.dict(os.environ, {MORNING_BRIEF_ENV: "09:00"}, clear=True):
            scheduler.ensure_env_morning_brief(now=frozen)

        original = runtime.store.reconcile_job_schedule_if_revision
        paused = [False]

        def pause_before_reconcile(*args, **kwargs):
            if not paused[0]:
                paused[0] = True
                scheduler.pause_job(MORNING_BRIEF_JOB_NAME)
            return original(*args, **kwargs)

        with mock.patch.object(runtime.store, "reconcile_job_schedule_if_revision", side_effect=pause_before_reconcile), \
             mock.patch.dict(os.environ, {MORNING_BRIEF_ENV: "10:15"}, clear=True):
            scheduler.ensure_env_morning_brief(now=frozen)
        row = dict(runtime.store.list_jobs()[0])
        if not paused[0] or int(row["enabled"]) != 0:
            raise SystemExit(f"concurrent environment reconciliation overwrote pause: {row}")
        if datetime.fromisoformat(str(row["next_run_at"])).strftime("%H:%M") != "09:00":
            raise SystemExit(f"concurrent pause accepted stale environment schedule update: {row}")

        scheduler.resume_job(MORNING_BRIEF_JOB_NAME)
        with mock.patch.dict(os.environ, {MORNING_BRIEF_ENV: ""}, clear=True):
            scheduler.ensure_env_morning_brief(now=frozen)
        env_disabled = dict(runtime.store.list_jobs()[0])
        env_disabled_revision = int(env_disabled["schedule_revision"])
        original_reconcile = runtime.store.reconcile_job_schedule_if_revision
        paused_disabled = [False]

        def pause_disabled_before_reconcile(*args, **kwargs):
            if not paused_disabled[0]:
                paused_disabled[0] = True
                scheduler.pause_job(MORNING_BRIEF_JOB_NAME)
            return original_reconcile(*args, **kwargs)

        with mock.patch.object(
            runtime.store,
            "reconcile_job_schedule_if_revision",
            side_effect=pause_disabled_before_reconcile,
        ), mock.patch.dict(os.environ, {MORNING_BRIEF_ENV: "10:15"}, clear=True):
            scheduler.ensure_env_morning_brief(now=frozen)
        paused_env_row = dict(runtime.store.list_jobs()[0])
        paused_env_metadata = json.loads(paused_env_row["metadata"] or "{}")
        if not paused_disabled[0] or int(paused_env_row["enabled"]) != 0:
            raise SystemExit(f"env-disabled pause race re-enabled Morning Brief: {paused_env_row}")
        if paused_env_metadata.get("disabled_by_env") is True:
            raise SystemExit(f"manual pause did not clear env-disable provenance: {paused_env_row}")
        if int(paused_env_row["schedule_revision"]) <= env_disabled_revision:
            raise SystemExit(f"metadata-only manual pause did not advance schedule revision: {paused_env_row}")


def test_resume_overdue_morning_brief_recalculates_future_occurrence() -> None:
    with TemporaryDirectory(prefix="jarvis-morning-brief-overdue-resume-") as temp:
        runtime = make_temp_runtime(Path(temp))
        scheduler = Scheduler(runtime.store, runtime.vault, runtime.config)
        frozen = datetime(2026, 7, 3, 10, 0)
        with mock.patch.dict(os.environ, {MORNING_BRIEF_ENV: "09:00"}, clear=True):
            scheduler.ensure_env_morning_brief(now=datetime(2026, 7, 2, 8, 0))
        enabled_before = dict(runtime.store.list_jobs()[0])
        scheduler.resume_job(MORNING_BRIEF_JOB_NAME)
        enabled_after = dict(runtime.store.list_jobs()[0])
        for key in ("enabled", "next_run_at", "schedule_revision", "metadata"):
            if enabled_after[key] != enabled_before[key]:
                raise SystemExit(
                    f"redundant Morning Brief resume mutated {key}: {enabled_before} -> {enabled_after}"
                )
        scheduler.pause_job(MORNING_BRIEF_JOB_NAME)
        scheduler.resume_job(MORNING_BRIEF_JOB_NAME)
        with mock.patch.dict(os.environ, {MORNING_BRIEF_ENV: "09:00"}, clear=True):
            scheduler.ensure_env_morning_brief(now=frozen)
        row = dict(runtime.store.list_jobs()[0])
        next_run = datetime.fromisoformat(str(row["next_run_at"]))
        if next_run <= frozen or next_run.strftime("%Y-%m-%d %H:%M") != "2026-07-04 09:00":
            raise SystemExit(f"overdue Morning Brief resume did not move to a future occurrence: {row}")


def test_due_digest_jobs_deliver_to_owner_telegram() -> None:
    """Digest jobs (weekly review, goal nudge, …) must push their output to
    the owner's Telegram when the ticker fires them — not just print to the
    daemon log. Delivery failure must not block the reschedule."""
    with TemporaryDirectory(prefix="jarvis-digest-delivery-") as temp:
        runtime = make_temp_runtime(Path(temp))
        scheduler = Scheduler(runtime.store, runtime.vault, runtime.config)
        past = iso(datetime.now() - timedelta(minutes=5))
        runtime.store.upsert_job("Goal Nudge", 1440, "goal_nudge", past)
        sent: list[str] = []
        with mock.patch.object(
            scheduler_module, "_send_owner_telegram", side_effect=lambda text: sent.append(text) or _accepted()
        ), mock.patch.object(
            scheduler_module,
            "build_goal_nudge",
            return_value=_goal_nudge_build(
                runtime,
                "## Goals Quiet\n- #1 Ship Jarvis | next: wire delivery",
            ),
        ):
            output = scheduler.run_due_jobs()
        if len(sent) != 1 or not sent[0].startswith("🎯 Goal Nudge"):
            raise SystemExit(f"due goal_nudge should deliver one titled Telegram message: {sent}")
        if "[accepted by Telegram API]" not in output:
            raise SystemExit(f"ticker output should note Telegram API acceptance: {output}")
        if "delivered to Telegram" in output:
            raise SystemExit(f"ticker output should not overclaim phone delivery: {output}")
        goal_row = [r for r in runtime.store.list_jobs() if r["name"] == "Goal Nudge"][0]
        goal_metadata = json.loads(goal_row["metadata"] or "{}")
        if goal_metadata.get("last_run_status") != "ok":
            raise SystemExit(f"successful digest run should record ok run status: {goal_metadata}")
        if "Ship Jarvis" in json.dumps(goal_metadata.get("run_history")):
            raise SystemExit(f"digest run history must not store job output content: {goal_metadata}")

        # Failure path: send fails, job must still be rescheduled to the future.
        runtime.store.upsert_job("Weekly Review", 10080, "weekly_review", past)
        with mock.patch.object(
            scheduler_module, "_send_owner_telegram", return_value={"ok": False, "error": "network_down"}
        ):
            failed = scheduler.run_due_jobs()
        for fragment in ["delivery failed:", "Telegram delivery failed", "setup check", "channel status", "network_down"]:
            if fragment not in failed:
                raise SystemExit(f"failed delivery should include actionable guidance {fragment!r}: {failed}")
        review_row = [r for r in runtime.store.list_jobs() if r["name"] == "Weekly Review"][0]
        if datetime.fromisoformat(str(review_row["next_run_at"])) <= datetime.now():
            raise SystemExit("failed delivery must not block rescheduling the job")
        review_metadata = json.loads(review_row["metadata"] or "{}")
        if review_metadata.get("last_run_status") != "failed":
            raise SystemExit(f"failed digest delivery should record failed run status: {review_metadata}")

        runtime.store.upsert_job("Weekly Review Redacted Error", 10080, "weekly_review", past)
        with mock.patch.object(
            scheduler_module,
            "_send_owner_telegram",
            return_value={"ok": False, "error": "/\x55sers/example/private SHOULD NOT APPEAR"},
        ):
            redacted = scheduler.run_due_jobs()
        for fragment in ["delivery failed:", "Telegram delivery failed", "setup check", "channel status"]:
            if fragment not in redacted:
                raise SystemExit(f"redacted digest delivery failure missed {fragment!r}: {redacted}")
        for forbidden in ["/Users", "SHOULD NOT APPEAR", "operator/private"]:
            if forbidden in redacted:
                raise SystemExit(f"redacted digest delivery failure leaked diagnostic {forbidden!r}: {redacted}")

        # Manual run_job_now must NOT double-deliver (its output goes back to the caller).
        sent.clear()
        with mock.patch.object(
            scheduler_module, "_send_owner_telegram", side_effect=lambda text: sent.append(text) or _accepted()
        ):
            scheduler.run_job_now("Goal Nudge")
        if sent:
            raise SystemExit(f"run_job_now should return output to the caller, not re-send to Telegram: {sent}")

        # Quiet output (nothing new to report) must skip delivery entirely.
        sent.clear()
        runtime.store.upsert_job("Goal Nudge", 1440, "goal_nudge", past)
        with mock.patch.object(
            scheduler_module, "_send_owner_telegram", side_effect=lambda text: sent.append(text) or _accepted()
        ), mock.patch.object(
            scheduler_module,
            "build_goal_nudge",
            return_value=_goal_nudge_build(
                runtime,
                "## Goals\n- No stale active goals.",
            ),
        ):
            quiet = scheduler.run_due_jobs()
        if sent:
            raise SystemExit(f"quiet goal nudge must not send an empty digest: {sent}")
        if "quiet — nothing new" not in quiet:
            raise SystemExit(f"ticker output should note the skipped quiet delivery: {quiet}")


def main() -> None:
    original_owner = os.environ.get("JARVIS_OWNER_TELEGRAM")
    os.environ["JARVIS_OWNER_TELEGRAM"] = "owner-smoke"
    try:
        test_parse_and_route_morning_brief_schedule()
        test_parse_and_route_on_demand_morning_brief_aliases()
        test_schedule_morning_brief_tool_is_idempotent()
        test_concurrent_morning_brief_schedule_and_metadata_are_atomic()
        test_morning_brief_schedule_metadata_failure_rolls_back()
        test_env_refresh_revalidates_schedule_before_metadata_merge()
        test_due_morning_brief_sends_once_to_owner_channel()
        test_morning_brief_delivery_failures_are_actionable_and_scrubbed()
        test_missing_owner_preflight_does_not_fence_same_day_morning_brief_retry()
        test_legacy_same_day_guard_prevents_upgrade_duplicate()
        test_on_demand_morning_brief_runtime_sends_once_to_owner_channel()
        test_on_demand_phone_brief_alias_sends_once_to_owner_channel()
        test_not_due_and_env_opt_in_do_not_send()
        test_env_default_legacy_and_empty_disable()
        test_env_morning_brief_normalizes_malformed_enabled_row()
        test_env_reconciliation_preserves_explicit_morning_brief_pause()
        test_env_reconciliation_cannot_overwrite_concurrent_pause()
        test_resume_overdue_morning_brief_recalculates_future_occurrence()
        test_due_digest_jobs_deliver_to_owner_telegram()
    finally:
        if original_owner is None:
            os.environ.pop("JARVIS_OWNER_TELEGRAM", None)
        else:
            os.environ["JARVIS_OWNER_TELEGRAM"] = original_owner
    print("Morning brief scheduler smoke passed")


if __name__ == "__main__":
    main()
