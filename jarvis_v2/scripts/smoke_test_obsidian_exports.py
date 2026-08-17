from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Barrier, Event, Thread, current_thread
from unittest.mock import patch

from jarvis_v2.config import JarvisConfig
from jarvis_v2.memory import obsidian as obsidian_module
from jarvis_v2.memory.obsidian import ObsidianVault
from jarvis_v2.memory.store import (
    DecisionRecord,
    GoalRecord,
    MemoryRecord,
    MemoryStore,
    PersonRecord,
    PreferenceRecord,
    SkillRecord,
    TaskRecord,
)
from jarvis_v2.scripts.test_runtime import make_temp_runtime


def _assert_redacted_note_tree(root: Path) -> None:
    files = list(root.rglob("*.md"))
    if not files:
        raise SystemExit("Expected Obsidian export notes to be written.")
    text = "\n".join(path.read_text(encoding="utf-8") for path in files)
    relative_names = "\n".join(path.relative_to(root).as_posix() for path in files)
    if "/\x55sers/" in text or "/private/" in text:
        raise SystemExit(f"Obsidian export note content leaked a raw local path:\n{text}")
    if "/\x55sers/" in relative_names or "/private/" in relative_names:
        raise SystemExit(f"Obsidian export note filename leaked a raw local path:\n{relative_names}")
    if "<local-path>" not in text and "local-path" not in relative_names:
        raise SystemExit("Expected Obsidian exports to include redacted legacy path markers.")


def _expect_containment_rejection(label: str, action) -> None:
    try:
        action()
    except ValueError as exc:
        if "within the Jarvis vault" not in str(exc):
            raise SystemExit(f"{label} used the wrong containment diagnostic: {exc}")
    else:
        raise SystemExit(f"{label} escaped the Jarvis vault.")


def _assert_all_write_paths_reject_symlink_escapes(base: Path) -> None:
    outside = base / "outside"
    outside.mkdir(parents=True)
    vault = ObsidianVault(base / "Vault")
    vault.init()

    skills = vault.root_path / "Skills"
    skills.rmdir()
    skills.symlink_to(outside, target_is_directory=True)
    _expect_containment_rejection(
        "Atomic write through a symlinked parent",
        lambda: vault.write_skill("escaped skill", "trigger", "body"),
    )
    if (outside / "escaped skill.md").exists():
        raise SystemExit("Rejected atomic parent escape created an outside note.")

    inbox_target = outside / "outside-inbox.md"
    inbox_target.write_text("outside inbox sentinel\n", encoding="utf-8")
    inbox = vault.root_path / "Inbox.md"
    inbox.unlink()
    inbox.symlink_to(inbox_target)
    _expect_containment_rejection("Atomic write to a symlink destination", vault.reset_inbox)
    if inbox_target.read_text(encoding="utf-8") != "outside inbox sentinel\n":
        raise SystemExit("Rejected atomic destination escape changed the outside note.")

    daily = vault.root_path / "Daily"
    daily.rmdir()
    daily.symlink_to(outside, target_is_directory=True)
    _expect_containment_rejection(
        "Daily append through a symlinked parent",
        lambda: vault.append_daily("escaped heading", "escaped body"),
    )
    if any(path.name.startswith("20") for path in outside.glob("*.md")):
        raise SystemExit("Rejected daily parent escape created an outside note.")

    profile_target = outside / "outside-profile.md"
    profile_target.write_text("outside profile sentinel\n", encoding="utf-8")
    profile = vault.root_path / "Profile.md"
    profile.unlink()
    profile.symlink_to(profile_target)
    _expect_containment_rejection(
        "Profile append to a symlink destination",
        lambda: vault.append_profile("escaped heading", "escaped body"),
    )
    if profile_target.read_text(encoding="utf-8") != "outside profile sentinel\n":
        raise SystemExit("Rejected profile destination escape changed the outside note.")


def _assert_concurrent_first_appends_are_preserved(base: Path) -> None:
    vault = ObsidianVault(base / "Concurrent Vault")
    vault.init()
    barrier = Barrier(3)
    entries = (("parallel alpha", "alpha body"), ("parallel beta", "beta body"))

    def append_entry(entry: tuple[str, str]) -> None:
        barrier.wait()
        vault.append_daily(*entry)

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(append_entry, entry) for entry in entries]
        barrier.wait()
        for future in futures:
            future.result()

    day = next((vault.root_path / "Daily").glob("*.md"))
    text = day.read_text(encoding="utf-8")
    if text.count(f"# {day.stem}") != 1:
        raise SystemExit(f"Concurrent first daily appends did not preserve one header:\n{text}")
    if any(text.count(value) != 1 for entry in entries for value in entry):
        raise SystemExit(f"Concurrent first daily appends lost or duplicated an entry:\n{text}")


def _assert_atomic_source_write_resists_symlink_swap(base: Path) -> None:
    outside = base / "outside"
    outside.mkdir(parents=True)
    outside_target = outside / "source-target.md"
    outside_target.write_text("outside source sentinel\n", encoding="utf-8")

    vault = ObsidianVault(base / "Vault")
    vault.init()
    destination = vault.root_path / "Sources" / "Swapped Source.md"
    real_replace = os.replace
    replace_calls = 0

    def swap_destination_then_replace(src, dst, *, src_dir_fd=None, dst_dir_fd=None):
        nonlocal replace_calls
        replace_calls += 1
        destination.symlink_to(outside_target)
        return real_replace(src, dst, src_dir_fd=src_dir_fd, dst_dir_fd=dst_dir_fd)

    with patch("jarvis_v2.memory.obsidian.os.replace", side_effect=swap_destination_then_replace):
        written = vault.write_source("Swapped Source", "contained source body")

    if replace_calls != 1:
        raise SystemExit(f"Source note write should use one atomic replacement, got {replace_calls}.")
    if outside_target.read_text(encoding="utf-8") != "outside source sentinel\n":
        raise SystemExit("Atomic source write followed a swapped destination symlink.")
    if written.is_symlink() or written.read_text(encoding="utf-8") != "contained source body":
        raise SystemExit("Atomic source write did not replace the swapped symlink with the intended note.")


def _assert_goal_paths_are_race_safe(base: Path) -> None:
    runtime = make_temp_runtime(base)
    store = runtime.store
    vault = runtime.vault
    goal_one = store.create_goal(GoalRecord("Same  title", purpose="first concurrent goal"))
    goal_two = store.create_goal(GoalRecord("Same title", purpose="second concurrent goal"))
    legacy_path = vault.root_path / "Projects" / "Same title.md"
    store_identity = store.get_store_identity()

    def export(goal_id: int) -> Path:
        return vault.write_goal(
            store.get_goal(goal_id),
            [],
            store_identity=store_identity,
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        future_one = pool.submit(export, goal_one)
        future_two = pool.submit(export, goal_two)
        path_one = future_one.result()
        path_two = future_two.result()

    expected_one = vault.root_path / "Projects" / f"Same title [{goal_one}-{store_identity}].md"
    expected_two = vault.root_path / "Projects" / f"Same title [{goal_two}-{store_identity}].md"
    if path_one != expected_one or path_two != expected_two or legacy_path.exists():
        raise SystemExit(f"Concurrent same-title goals did not use stable ID-bearing paths: {path_one}, {path_two}")
    text_one = path_one.read_text(encoding="utf-8")
    text_two = path_two.read_text(encoding="utf-8")
    if f"id: {goal_one}" not in text_one or "first concurrent goal" not in text_one:
        raise SystemExit(f"First concurrent goal note lost its identity or content:\n{text_one}")
    if f"id: {goal_two}" not in text_two or "second concurrent goal" not in text_two:
        raise SystemExit(f"Second concurrent goal note lost its identity or content:\n{text_two}")
    if export(goal_one) != path_one or export(goal_two) != path_two:
        raise SystemExit("Re-exporting same-title goals changed their ID-bearing paths.")

    legacy_goal = store.create_goal(GoalRecord("Legacy match", purpose="updated legacy content"))
    matching_legacy = vault.root_path / "Projects" / "Legacy match.md"
    matching_legacy.write_text(
        f"---\nid: {legacy_goal}\nstatus: active\n---\n\n# Legacy match\n\nold content\n",
        encoding="utf-8",
    )
    legacy_before = matching_legacy.read_bytes()
    reused_path = export(legacy_goal)
    expected_legacy_projection = (
        vault.root_path / "Projects" / f"Legacy match [{legacy_goal}-{store_identity}].md"
    )
    if (
        reused_path != expected_legacy_projection
        or matching_legacy.read_bytes() != legacy_before
        or "updated legacy content" not in reused_path.read_text(encoding="utf-8")
    ):
        raise SystemExit("An ambiguous legacy goal note was not preserved beside the owned projection.")

    unproven_goal = store.create_goal(GoalRecord("Legacy unproven", purpose="new ID-bearing content"))
    unproven_legacy = vault.root_path / "Projects" / "Legacy unproven.md"
    unproven_legacy.write_text(
        f"# Legacy unproven\n\nAn ordinary body line must not prove identity.\nid: {unproven_goal}\n",
        encoding="utf-8",
    )
    unproven_path = export(unproven_goal)
    expected_unproven = (
        vault.root_path / "Projects" / f"Legacy unproven [{unproven_goal}-{store_identity}].md"
    )
    if unproven_path != expected_unproven or "ordinary body line" not in unproven_legacy.read_text(encoding="utf-8"):
        raise SystemExit("An unproven title-only note was incorrectly reused as a goal mirror.")

    collision_goal = store.create_goal(GoalRecord("Owned collision", purpose="must not overwrite"))
    collision_path = (
        vault.root_path / "Projects" / f"Owned collision [{collision_goal}-{store_identity}].md"
    )
    collision_path.write_text("# ordinary user note\n", encoding="utf-8")
    try:
        export(collision_goal)
    except FileExistsError:
        pass
    else:
        raise SystemExit("An unowned ID-bearing goal destination was overwritten.")
    if collision_path.read_text(encoding="utf-8") != "# ordinary user note\n":
        raise SystemExit("Goal collision refusal changed the unowned destination.")


def _assert_task_mirror_ordering(base: Path) -> None:
    runtime = make_temp_runtime(base)
    store = runtime.store
    vault = runtime.vault
    second_store = MemoryStore(store.db_path)
    second_vault = ObsidianVault(vault.vault_path, vault.root)

    first_id = store.add_task(TaskRecord("first mirror task"))
    vault.sync_open_tasks(store)

    older_started = Event()
    resume_older = Event()
    older_errors: list[BaseException] = []
    real_sidecar_lock = obsidian_module._exclusive_sidecar_lock

    @contextmanager
    def pause_older_before_lock(root: Path, path: Path):
        if current_thread().name == "older-task-mirror":
            older_started.set()
            if not resume_older.wait(5):
                raise RuntimeError("older task mirror caller was not resumed")
        with real_sidecar_lock(root, path):
            yield

    def delayed_older_caller() -> None:
        try:
            second_vault.sync_open_tasks(second_store)
        except BaseException as exc:
            older_errors.append(exc)

    with patch("jarvis_v2.memory.obsidian._exclusive_sidecar_lock", side_effect=pause_older_before_lock):
        older_thread = Thread(target=delayed_older_caller, name="older-task-mirror")
        older_thread.start()
        if not older_started.wait(5):
            raise SystemExit("Older task mirror caller did not reach the deterministic pause.")
        second_id = store.add_task(TaskRecord("newer mirror task"))
        vault.sync_open_tasks(store)
        resume_older.set()
        older_thread.join(5)
    if older_thread.is_alive() or older_errors:
        raise SystemExit(f"Older task mirror caller did not finish cleanly: {older_errors}")
    text = (vault.root_path / "Tasks" / "Open Tasks.md").read_text(encoding="utf-8")
    if f"#{first_id} first mirror task" not in text or f"#{second_id} newer mirror task" not in text:
        raise SystemExit(f"An older caller resuming last overwrote the canonical task projection:\n{text}")

    publication_started = Event()
    release_publication = Event()
    mutation_finished = Event()
    concurrent_errors: list[BaseException] = []
    real_replace_text = obsidian_module._replace_text
    replace_calls = 0

    def pause_first_task_publication(root: Path, path: Path, content: str) -> None:
        nonlocal replace_calls
        if path.name == "Open Tasks.md":
            replace_calls += 1
            if replace_calls == 1:
                publication_started.set()
                if not release_publication.wait(5):
                    raise RuntimeError("task mirror publication was not released")
        real_replace_text(root, path, content)

    def publish_snapshot() -> None:
        try:
            vault.sync_open_tasks(store)
        except BaseException as exc:
            concurrent_errors.append(exc)

    during_id: list[int] = []

    def mutate_then_publish() -> None:
        try:
            during_id.append(second_store.add_task(TaskRecord("mutation during publication")))
            mutation_finished.set()
            second_vault.sync_open_tasks(second_store)
        except BaseException as exc:
            concurrent_errors.append(exc)

    with patch("jarvis_v2.memory.obsidian._replace_text", side_effect=pause_first_task_publication):
        publisher = Thread(target=publish_snapshot)
        publisher.start()
        if not publication_started.wait(5):
            raise SystemExit("Task mirror did not reach the publication pause.")
        mutator = Thread(target=mutate_then_publish)
        mutator.start()
        if mutation_finished.wait(0.2):
            raise SystemExit("A task mutation committed between canonical snapshot capture and publication.")
        release_publication.set()
        publisher.join(5)
        mutator.join(5)
    if publisher.is_alive() or mutator.is_alive() or not mutation_finished.is_set() or not during_id or concurrent_errors:
        raise SystemExit(f"Cross-instance task mirror ordering test did not finish cleanly: {concurrent_errors}")
    text = (vault.root_path / "Tasks" / "Open Tasks.md").read_text(encoding="utf-8")
    if f"#{during_id[0]} mutation during publication" not in text:
        raise SystemExit(f"The post-publication mutation was not repaired by its canonical sync:\n{text}")


def _assert_task_mirror_repairs_after_write_failure(base: Path) -> None:
    runtime = make_temp_runtime(base)
    store = runtime.store
    vault = runtime.vault
    body = "database remains authoritative after mirror failure"
    with patch("jarvis_v2.memory.obsidian._replace_text", side_effect=OSError("task mirror failed")):
        try:
            runtime.registry.get("add_task").handler({"body": body})
        except OSError as exc:
            if "task mirror failed" not in str(exc):
                raise SystemExit(f"Task mirror write surfaced the wrong failure: {exc}")
        else:
            raise SystemExit("Expected the task mirror write to fail.")
    rows = store.list_tasks(status="open", limit=10)
    matching = [row for row in rows if row["body"] == body]
    if len(matching) != 1:
        raise SystemExit("A failed task mirror write rolled back or duplicated the authoritative database task.")
    vault.sync_open_tasks(store)
    text = (vault.root_path / "Tasks" / "Open Tasks.md").read_text(encoding="utf-8")
    if f"#{matching[0]['id']} {body}" not in text:
        raise SystemExit(f"A later canonical sync did not repair the failed task mirror:\n{text}")


def main() -> None:
    with TemporaryDirectory(prefix="jarvis-obsidian-exports-") as temp:
        temp_path = Path(temp)
        _assert_all_write_paths_reject_symlink_escapes(temp_path / "containment")
        _assert_concurrent_first_appends_are_preserved(temp_path / "concurrency")
        _assert_atomic_source_write_resists_symlink_swap(temp_path / "source-swap")
        _assert_goal_paths_are_race_safe(temp_path / "goal-race")
        _assert_task_mirror_ordering(temp_path / "task-mirror-ordering")
        _assert_task_mirror_repairs_after_write_failure(temp_path / "task-mirror-repair")

        runtime = make_temp_runtime(temp_path)
        store = runtime.store
        vault = runtime.vault

        inbox_path = vault.root_path / "Inbox.md"
        original_inbox = inbox_path.read_text(encoding="utf-8")
        with patch("jarvis_v2.memory.obsidian.os.replace", side_effect=OSError("replace failed")):
            try:
                vault.reset_inbox()
            except OSError as exc:
                if "replace failed" not in str(exc):
                    raise SystemExit(f"Atomic replace surfaced the wrong failure: {exc}")
            else:
                raise SystemExit("Expected the mocked atomic replace to fail.")
        if inbox_path.read_text(encoding="utf-8") != original_inbox:
            raise SystemExit("Failed atomic replace changed the existing Inbox note.")
        if list(inbox_path.parent.glob(f".{inbox_path.name}.*.tmp")):
            raise SystemExit("Failed atomic replace left a temporary note behind.")

        store.add_memory(
            MemoryRecord(
                "/\x55sers/example/private/memory-category",
                "/\x55sers/example/private/memory-title",
                "legacy memory body /private/tmp/memory-body",
                source="/\x55sers/example/private/source",
            )
        )
        vault.write_memory(
            MemoryRecord(
                "/\x55sers/example/private/manual-memory-category",
                "/\x55sers/example/private/manual-memory-title",
                "manual memory body /private/tmp/manual-memory-body",
                source="/private/tmp/manual-memory-source",
            ),
            memory_id=99,
            store_identity=store.get_store_identity(),
        )

        vault.write_skill(
            "/\x55sers/example/private/skill-name",
            "trigger /private/tmp/skill-trigger",
            "skill procedure /\x55sers/example/private/skill-body",
            tags="/private/tmp/skill-tag",
        )

        legacy_goal_id = store.create_goal(
            GoalRecord(
                "/\x55sers/example/private/goal-title",
                purpose="/private/tmp/goal-purpose",
                horizon="/\x55sers/example/private/goal-horizon",
            )
        )
        store.add_goal_step(legacy_goal_id, "/private/tmp/goal-step")
        goal = store.get_goal(legacy_goal_id)
        vault.write_goal(
            goal,
            store.list_goal_steps(legacy_goal_id),
            store_identity=store.get_store_identity(),
        )

        store.add_task(
            TaskRecord(
                "/\x55sers/example/private/task-body",
                due="/private/tmp/task-due",
                priority="/\x55sers/example/private/task-priority",
            )
        )
        vault.sync_open_tasks(store)

        store.add_pending_approval(
            "/\x55sers/example/private/session",
            "/\x55sers/example/private/approval-input",
            "/private/tmp/approval-tool",
            "/\x55sers/example/private/approval-reason",
            planned_args={"/private/tmp/key": "/\x55sers/example/private/value"},
        )
        vault.write_pending_approvals(store.list_pending_approvals(limit=10))

        decision_id = store.add_decision(
            DecisionRecord(
                "/\x55sers/example/private/decision-title",
                rationale="/private/tmp/decision-rationale",
                impact="/\x55sers/example/private/decision-impact",
            )
        )
        vault.write_decision(store.get_decision(decision_id))

        person_id = store.upsert_person(
            PersonRecord(
                "/\x55sers/example/private/person-name",
                relation="/private/tmp/person-relation",
                notes="/\x55sers/example/private/person-notes",
            )
        )
        store.log_person_interaction(person_id, "/private/tmp/person-interaction", "/\x55sers/example/private/contact-time")
        vault.write_person(
            store.get_person(person_id=person_id),
            store.list_person_interactions(person_id),
            store_identity=store.get_store_identity(),
        )

        store.set_preference(
            PreferenceRecord(
                "/\x55sers/example/private/preference-key",
                "/private/tmp/preference-value",
                "/\x55sers/example/private/preference-category",
            )
        )
        vault.write_preferences(store.list_preferences(status="active", limit=10))

        vault.write_session("/\x55sers/example/private/session-id", "session body /private/tmp/session-body")
        vault.write_reflection("/\x55sers/example/private/reflection-title", "reflection body /private/tmp/reflection-body")
        vault.write_source("/\x55sers/example/private/source-title", "source body /private/tmp/source-body")
        vault.write_return_brief("return brief /\x55sers/example/private/return-body")
        vault.append_daily("/\x55sers/example/private/daily-heading", "daily body /private/tmp/daily-body")
        vault.append_profile("/private/tmp/profile-heading", "profile body /\x55sers/example/private/profile-body")
        profile_text = vault.read_profile(4000)
        if "/\x55sers/" in profile_text or "/private/" in profile_text or "<local-path>" not in profile_text:
            raise SystemExit(f"Profile read should redact direct legacy profile paths: {profile_text}")

        case_path = vault.write_execution_case(7, "/\x55sers/example/private/execution-request", "case body /private/tmp/case-body")
        vault.append_execution_case_event(str(case_path), "case event /\x55sers/example/private/case-event")

        append_barrier = Barrier(3)
        concurrent_events = ("concurrent event alpha", "concurrent event beta")

        def append_case_event(event_body: str) -> None:
            append_barrier.wait()
            vault.append_execution_case_event(str(case_path), event_body)

        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(append_case_event, event_body) for event_body in concurrent_events]
            append_barrier.wait()
            for future in futures:
                future.result()
        case_text = case_path.read_text(encoding="utf-8")
        if any(case_text.count(event_body) != 1 for event_body in concurrent_events):
            raise SystemExit(f"Concurrent execution-case appends did not preserve both events:\n{case_text}")

        outside_path = Path(temp) / "outside-case.md"
        try:
            vault.append_execution_case_event(str(outside_path), "must not escape")
        except ValueError as exc:
            if "within the Jarvis vault" not in str(exc):
                raise SystemExit(f"Outside execution-case path used the wrong diagnostic: {exc}")
        else:
            raise SystemExit("Execution-case event append escaped the Jarvis vault.")
        if outside_path.exists():
            raise SystemExit("Rejected execution-case event created an outside file.")

        outside_dir = Path(temp) / "outside-case-dir"
        outside_dir.mkdir()
        escape_link = vault.root_path / "Automations" / "escape-link"
        escape_link.symlink_to(outside_dir, target_is_directory=True)
        try:
            vault.append_execution_case_event(str(escape_link / "case.md"), "must not follow symlink")
        except ValueError:
            pass
        else:
            raise SystemExit("Execution-case event append followed a symlink outside the Jarvis vault.")
        if (outside_dir / "case.md").exists():
            raise SystemExit("Rejected symlink escape created an outside file.")

        _assert_redacted_note_tree(runtime.vault.root_path)

        config = JarvisConfig(
            data_dir=Path(temp),
            db_path=Path(temp) / "jarvis.sqlite",
            obsidian_vault=Path(temp) / "Vault",
            obsidian_root="Jarvis",
            use_model_planner=False,
        )
        if config.obsidian_root != "Jarvis":
            raise SystemExit("Config sanity check failed.")
        print("Obsidian export redaction smoke passed")


if __name__ == "__main__":
    main()
