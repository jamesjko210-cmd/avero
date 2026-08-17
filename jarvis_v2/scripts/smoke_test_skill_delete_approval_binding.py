from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from tempfile import TemporaryDirectory

from jarvis_v2.memory.store import MemoryStore, SkillRecord
from jarvis_v2.scripts.test_runtime import (
    approve_pending_runtime_approval,
    make_temp_runtime,
)


def _save(runtime, name: str, body: str = "body"):
    result = runtime.registry.get("save_skill").handler(
        {"name": name, "trigger": "test", "body": body}
    )
    if not result.ok:
        raise SystemExit(f"could not seed skill {name!r}: {result}")
    return result


def _queue(runtime, command: str) -> tuple[int, dict]:
    held = runtime.handle(command)
    approvals = [
        result
        for result in held.tool_results
        if result.tool_name == "delete_skill"
        and result.metadata.get("requires_confirmation") is True
    ]
    if len(approvals) != 1:
        raise SystemExit(f"delete command did not queue exactly one approval: {command!r} -> {held}")
    result = approvals[0]
    approval_id = result.metadata.get("approval_id")
    planned_args = result.metadata.get("planned_args")
    if type(approval_id) is not int or type(planned_args) is not dict:
        raise SystemExit(f"delete approval missed its stored binding: {result.metadata}")
    row = runtime.store.get_pending_approval(approval_id)
    if row is None or json.loads(row["planned_args"]) != planned_args:
        raise SystemExit("pending approval did not persist the exact bound arguments")
    return approval_id, planned_args


def _execute(runtime, command: str, approval_id: int):
    transition = approve_pending_runtime_approval(runtime, approval_id)
    if transition.metadata.get("rerun_user_input") != command:
        raise SystemExit("approval transition changed the stored delete request")
    return runtime.handle(command, approved=True, approved_approval_id=approval_id)


def _skill_rows(runtime) -> list[tuple]:
    with runtime.store.connect() as conn:
        return [
            tuple(row)
            for row in conn.execute(
                "SELECT id, name, trigger, body, tags, revision FROM skills ORDER BY id"
            )
        ]


def _skill_notes(runtime) -> dict[str, bytes]:
    root = runtime.vault.root_path / "Skills"
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in sorted(root.glob("*.md"))
    }


def _assert_bound_args(args: dict, *, expected_name: str) -> None:
    if set(args) != {
        "name",
        "target_skill_id",
        "target_skill_revision",
        "target_skill_name",
    }:
        raise SystemExit(f"delete approval stored the wrong argument shape: {args}")
    if (
        args["target_skill_name"] != expected_name
        or type(args["target_skill_id"]) is not int
        or args["target_skill_id"] < 1
        or type(args["target_skill_revision"]) is not int
        or args["target_skill_revision"] < 1
    ):
        raise SystemExit(f"delete approval stored a malformed immutable target: {args}")


def test_valid_binding_and_replay() -> None:
    with TemporaryDirectory(prefix="jarvis-delete-skill-bound-") as temp:
        runtime = make_temp_runtime(Path(temp))
        saved = _save(runtime, "Bound Target")
        command = "delete skill Bound Target"
        approval_id, args = _queue(runtime, command)
        _assert_bound_args(args, expected_name="Bound Target")
        if args["target_skill_id"] != saved.metadata["skill_id"]:
            raise SystemExit("delete approval bound the wrong skill ID")
        result = _execute(runtime, command, approval_id)
        if not result.verified or len(result.tool_results) != 1 or not result.tool_results[0].ok:
            raise SystemExit(f"valid bound delete did not complete: {result}")
        with runtime.store.connect() as conn:
            if conn.execute("SELECT 1 FROM skills WHERE id = ?", (args["target_skill_id"],)).fetchone():
                raise SystemExit("valid bound delete left its database row")
        replay = runtime.handle(command, approved=True, approved_approval_id=approval_id)
        if replay.plan.actions or any(item.metadata.get("handler_invoked") is True for item in replay.tool_results):
            raise SystemExit(f"completed delete approval replayed its handler: {replay}")


def test_same_id_update_is_stale() -> None:
    with TemporaryDirectory(prefix="jarvis-delete-skill-update-") as temp:
        runtime = make_temp_runtime(Path(temp))
        _save(runtime, "Mutable Target", "before")
        command = "delete skill Mutable Target"
        approval_id, args = _queue(runtime, command)
        _save(runtime, "Mutable Target", "after")
        before_rows = _skill_rows(runtime)
        before_notes = _skill_notes(runtime)
        result = _execute(runtime, command, approval_id)
        tool_result = result.tool_results[0]
        if tool_result.ok or tool_result.metadata.get("reason") != "stale_skill_binding":
            raise SystemExit(f"changed skill did not make approval stale: {result}")
        if _skill_rows(runtime) != before_rows or _skill_notes(runtime) != before_notes:
            raise SystemExit("stale same-ID approval changed skill state or projections")
        with runtime.store.connect() as conn:
            row = conn.execute("SELECT revision FROM skills WHERE id = ?", (args["target_skill_id"],)).fetchone()
        if row is None or int(row["revision"]) != args["target_skill_revision"] + 1:
            raise SystemExit("skill update did not advance the immutable revision")


def test_replacement_and_fuzzy_retarget_cannot_redirect() -> None:
    with TemporaryDirectory(prefix="jarvis-delete-skill-replace-") as temp:
        runtime = make_temp_runtime(Path(temp))
        _save(runtime, "Replacement Target")
        command = "delete skill Replacement Target"
        approval_id, args = _queue(runtime, command)
        removed = runtime.store.delete_skill_record("Replacement Target")
        replacement = _save(runtime, "Replacement Target", "replacement")
        if removed is None or replacement.metadata["skill_id"] == args["target_skill_id"]:
            raise SystemExit("replacement fixture did not create a new durable identity")
        before_rows = _skill_rows(runtime)
        before_notes = _skill_notes(runtime)
        result = _execute(runtime, command, approval_id)
        if result.tool_results[0].metadata.get("reason") != "stale_skill_binding":
            raise SystemExit(f"replacement target did not stale the approval: {result}")
        if _skill_rows(runtime) != before_rows or _skill_notes(runtime) != before_notes:
            raise SystemExit("stale replacement approval changed the replacement or notes")

    with TemporaryDirectory(prefix="jarvis-delete-skill-fuzzy-") as temp:
        runtime = make_temp_runtime(Path(temp))
        first = _save(runtime, "Original Fuzzy Needle")
        command = "delete skill Fuzzy Needle"
        approval_id, args = _queue(runtime, command)
        _assert_bound_args(args, expected_name="Original Fuzzy Needle")
        second = _save(runtime, "Later Fuzzy Needle")
        result = _execute(runtime, command, approval_id)
        if not result.tool_results[0].ok:
            raise SystemExit(f"unique fuzzy target did not remain bound after a decoy appeared: {result}")
        with runtime.store.connect() as conn:
            first_row = conn.execute("SELECT 1 FROM skills WHERE id = ?", (first.metadata["skill_id"],)).fetchone()
            second_row = conn.execute("SELECT 1 FROM skills WHERE id = ?", (second.metadata["skill_id"],)).fetchone()
        if first_row is not None or second_row is None:
            raise SystemExit("approved fuzzy deletion retargeted to the newer match")


def test_ambiguity_and_legacy_approval_fail_closed() -> None:
    with TemporaryDirectory(prefix="jarvis-delete-skill-ambiguous-") as temp:
        runtime = make_temp_runtime(Path(temp))
        _save(runtime, "Case Target")
        with runtime.store.connect() as conn:
            conn.execute(
                """
                INSERT INTO skills(
                    name, trigger, body, tags, review_status, created_at, updated_at
                )
                SELECT 'case target', trigger, body, tags, review_status, created_at, updated_at
                FROM skills WHERE name = 'Case Target'
                """
            )
        before_rows = _skill_rows(runtime)
        before_notes = _skill_notes(runtime)
        held = runtime.handle("delete skill CASE TARGET")
        if any(result.metadata.get("approval_id") for result in held.tool_results):
            raise SystemExit("ambiguous casefold target queued an approval")
        if not held.tool_results or held.tool_results[0].metadata.get("reason") != "ambiguous_target":
            raise SystemExit(f"ambiguous casefold target did not fail clearly: {held}")
        if _skill_rows(runtime) != before_rows or _skill_notes(runtime) != before_notes:
            raise SystemExit("ambiguous delete changed skill state")

        legacy = _save(runtime, "Legacy Approval")
        command = "delete skill Legacy Approval"
        approval_id = runtime.store.add_pending_approval(
            runtime.session_id,
            command,
            "delete_skill",
            "legacy unbound approval",
            {"name": "Legacy Approval"},
        )
        before_rows = _skill_rows(runtime)
        before_notes = _skill_notes(runtime)
        result = _execute(runtime, command, approval_id)
        tool_result = result.tool_results[0]
        if (
            tool_result.metadata.get("failure_kind") != "tool_arguments_invalid"
            or tool_result.metadata.get("handler_invoked") is not False
        ):
            raise SystemExit(f"legacy approval reached the destructive handler: {result}")
        if _skill_rows(runtime) != before_rows or _skill_notes(runtime) != before_notes:
            raise SystemExit("legacy name-only approval deleted or changed a skill")
        with runtime.store.connect() as conn:
            if conn.execute("SELECT 1 FROM skills WHERE id = ?", (legacy.metadata["skill_id"],)).fetchone() is None:
                raise SystemExit("legacy name-only approval removed its target")
            claim = conn.execute(
                "SELECT outcome FROM approval_execution_claims WHERE approval_id = ?",
                (approval_id,),
            ).fetchone()
        if claim is None or claim["outcome"] != "failed":
            raise SystemExit(f"legacy approval claim did not finalize as a bounded failure: {claim}")


def test_store_migration_and_concurrent_exact_delete() -> None:
    with TemporaryDirectory(prefix="jarvis-delete-skill-migration-") as temp:
        path = Path(temp) / "legacy.sqlite"
        with sqlite3.connect(path) as conn:
            conn.execute(
                """
                CREATE TABLE skills (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL UNIQUE,
                    trigger TEXT NOT NULL DEFAULT '',
                    body TEXT NOT NULL,
                    tags TEXT NOT NULL DEFAULT '',
                    success_count INTEGER NOT NULL DEFAULT 0,
                    failure_count INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                "INSERT INTO skills(name, trigger, body, tags, created_at, updated_at) "
                "VALUES ('Legacy Row', '', 'body', '', '2026-01-01T00:00:00Z', '2026-01-01T00:00:00Z')"
            )
        store = MemoryStore(path)
        store.init()
        resolution = store.resolve_skill_delete_target("Legacy Row")
        if resolution.status != "resolved" or resolution.revision != 1:
            raise SystemExit(f"legacy migration did not initialize skill revision 1: {resolution}")

    with TemporaryDirectory(prefix="jarvis-delete-skill-concurrent-") as temp:
        store = MemoryStore(Path(temp) / "skills.sqlite")
        store.init()
        store.save_skill(SkillRecord("Concurrent Target", "", "body"))
        target = store.resolve_skill_delete_target("Concurrent Target")
        if target.status != "resolved":
            raise SystemExit("concurrent delete fixture did not resolve")
        statuses: list[str] = []
        errors: list[BaseException] = []

        def run_delete() -> None:
            try:
                statuses.append(
                    store.delete_skill_exact(
                        int(target.skill_id), int(target.revision), target.name
                    ).status
                )
            except BaseException as exc:
                errors.append(exc)

        threads = [threading.Thread(target=run_delete) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5)
        if errors or any(thread.is_alive() for thread in threads):
            raise SystemExit(f"concurrent exact delete failed: {errors}")
        if sorted(statuses) != ["deleted", "not_found"]:
            raise SystemExit(f"concurrent exact delete did not have one winner: {statuses}")


def main() -> None:
    test_valid_binding_and_replay()
    test_same_id_update_is_stale()
    test_replacement_and_fuzzy_retarget_cannot_redirect()
    test_ambiguity_and_legacy_approval_fail_closed()
    test_store_migration_and_concurrent_exact_delete()
    print("Skill delete approval binding smoke passed")


if __name__ == "__main__":
    main()
