from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from queue import Queue
from tempfile import TemporaryDirectory
from threading import Event, Thread
from unittest import mock

from jarvis_v2.automations import scheduler as scheduler_module
from jarvis_v2.automations.scheduler import Scheduler, iso
from jarvis_v2.scripts.test_runtime import make_temp_runtime


JOB_TYPE = "mock_pause_effect"


def _due_job(runtime, name: str):
    job_id = runtime.store.upsert_job(
        name,
        60,
        JOB_TYPE,
        iso(datetime.now() - timedelta(minutes=5)),
    )
    return next(row for row in runtime.store.list_jobs() if int(row["id"]) == job_id)


def _job(runtime, job_id: int):
    return next(row for row in runtime.store.list_jobs() if int(row["id"]) == job_id)


def _run_due_without_external_io(scheduler: Scheduler) -> str:
    with mock.patch.object(
        Scheduler,
        "ensure_env_morning_brief",
        return_value=None,
    ), mock.patch.object(
        scheduler_module,
        "_send_owner_telegram",
        side_effect=AssertionError("pause smoke must not send"),
    ):
        return scheduler.run_due_jobs()


def _assert_completed_once(row, *, original_next_run: str, label: str) -> None:
    if row["last_run_at"] is None:
        raise SystemExit(f"{label} did not record the completed occurrence")
    if datetime.fromisoformat(str(row["next_run_at"])) <= datetime.fromisoformat(original_next_run):
        raise SystemExit(f"{label} did not advance the completed occurrence: {dict(row)}")
    if row["active_occurrence_key"] is not None:
        raise SystemExit(f"{label} left replayable occurrence state behind: {dict(row)}")
    if row["lease_token"] is not None or row["lease_expires_at"] is not None:
        raise SystemExit(f"{label} left active lease state behind: {dict(row)}")


def test_post_effect_pause_owner_finalizes_and_resume_does_not_replay() -> None:
    with TemporaryDirectory(prefix="jarvis-scheduler-post-effect-pause-") as temp:
        runtime = make_temp_runtime(Path(temp))
        name = "Post Effect Pause"
        original = _due_job(runtime, name)
        job_id = int(original["id"])
        original_next_run = str(original["next_run_at"])
        scheduler = Scheduler(runtime.store, runtime.vault, runtime.config)
        effects: list[str] = []
        pause_results: list[str] = []
        effect_published = Event()
        release_handler = Event()
        tick_result: Queue[object] = Queue()

        def publish_then_pause(job_type: str, **_kwargs) -> str:
            if job_type != JOB_TYPE:
                raise AssertionError(f"unexpected mocked job type: {job_type}")
            effects.append("published")
            effect_published.set()
            if not release_handler.wait(timeout=5):
                raise RuntimeError("post-effect pause smoke timed out")
            return "mock effect published before pause"

        def run_tick() -> None:
            try:
                tick_result.put(_run_due_without_external_io(scheduler))
            except BaseException as exc:
                tick_result.put(exc)

        with mock.patch.object(scheduler, "_run_job_type", side_effect=publish_then_pause):
            worker = Thread(target=run_tick, name="scheduler-post-effect-pause-smoke")
            worker.start()
            try:
                if not effect_published.wait(timeout=5):
                    raise SystemExit("scheduled handler did not publish its mocked effect")
                pause_results.append(scheduler.pause_job(name))
            finally:
                release_handler.set()
                worker.join(timeout=5)
            if worker.is_alive():
                raise SystemExit("post-effect pause worker did not terminate")
            first_tick = tick_result.get_nowait()
            if isinstance(first_tick, BaseException):
                raise SystemExit(f"post-effect pause worker raised unexpectedly: {first_tick!r}")
            paused = _job(runtime, job_id)
            if effects != ["published"] or pause_results != [f"Paused {name}."]:
                raise SystemExit(
                    f"post-effect pause fixture did not reach its exact boundary: "
                    f"effects={effects!r} pauses={pause_results!r}"
                )
            if f"Ran {name}:" not in first_tick or "Skipped stale claim" in first_tick:
                raise SystemExit(f"post-effect pause discarded the owner's completion: {first_tick!r}")
            if int(paused["enabled"]) != 0:
                raise SystemExit(f"post-effect pause did not keep the job disabled: {dict(paused)}")
            _assert_completed_once(
                paused,
                original_next_run=original_next_run,
                label="post-effect pause",
            )

            if scheduler.resume_job(name) != f"Resumed {name}.":
                raise SystemExit("post-effect fixture could not resume the completed job")
            resumed_tick = _run_due_without_external_io(scheduler)

        resumed = _job(runtime, job_id)
        if int(resumed["enabled"]) != 1:
            raise SystemExit(f"resume did not enable the finalized job: {dict(resumed)}")
        if resumed_tick != "No jobs due." or effects != ["published"]:
            raise SystemExit(
                f"resume replayed the already-published occurrence: "
                f"tick={resumed_tick!r} effects={effects!r}"
            )


def test_pre_effect_pause_blocks_claim_until_resume() -> None:
    with TemporaryDirectory(prefix="jarvis-scheduler-pre-effect-pause-") as temp:
        runtime = make_temp_runtime(Path(temp))
        name = "Pre Effect Pause"
        original = _due_job(runtime, name)
        job_id = int(original["id"])
        original_next_run = str(original["next_run_at"])
        scheduler = Scheduler(runtime.store, runtime.vault, runtime.config)
        effects: list[str] = []

        if scheduler.pause_job(name) != f"Paused {name}.":
            raise SystemExit("pre-effect fixture could not pause the due job")

        def publish(job_type: str, **_kwargs) -> str:
            if job_type != JOB_TYPE:
                raise AssertionError(f"unexpected mocked job type: {job_type}")
            effects.append("published")
            return "mock effect published after resume"

        with mock.patch.object(scheduler, "_run_job_type", side_effect=publish) as handler:
            paused_tick = _run_due_without_external_io(scheduler)
            paused = _job(runtime, job_id)
            if paused_tick != "No jobs due." or handler.call_count != 0 or effects:
                raise SystemExit(
                    f"pre-effect pause allowed a new claim: "
                    f"tick={paused_tick!r} calls={handler.call_count} effects={effects!r}"
                )
            if (
                int(paused["enabled"]) != 0
                or paused["last_run_at"] is not None
                or paused["active_occurrence_key"] is not None
                or paused["lease_token"] is not None
            ):
                raise SystemExit(f"pre-effect pause mutated claim/run state: {dict(paused)}")
            if scheduler._claim_job(paused, due_before=iso(datetime.now())) is not None:
                raise SystemExit("disabled due job remained directly claimable")

            if scheduler.resume_job(name) != f"Resumed {name}.":
                raise SystemExit("pre-effect fixture could not resume the untouched occurrence")
            resumed_tick = _run_due_without_external_io(scheduler)
            replay_tick = _run_due_without_external_io(scheduler)

        completed = _job(runtime, job_id)
        if f"Ran {name}:" not in resumed_tick or effects != ["published"]:
            raise SystemExit(
                f"resume did not run the untouched occurrence exactly once: "
                f"tick={resumed_tick!r} effects={effects!r}"
            )
        if replay_tick != "No jobs due." or handler.call_count != 1:
            raise SystemExit(
                f"completed resumed occurrence replayed: "
                f"tick={replay_tick!r} calls={handler.call_count}"
            )
        _assert_completed_once(
            completed,
            original_next_run=original_next_run,
            label="pre-effect resume",
        )


def main() -> None:
    test_post_effect_pause_owner_finalizes_and_resume_does_not_replay()
    test_pre_effect_pause_blocks_claim_until_resume()
    print("Scheduler in-flight pause smoke passed")


if __name__ == "__main__":
    main()
