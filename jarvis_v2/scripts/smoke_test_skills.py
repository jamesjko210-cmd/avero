from __future__ import annotations

import threading
from pathlib import Path
from tempfile import TemporaryDirectory

from jarvis_v2.agent.planner import RuleBasedPlanner
from jarvis_v2.agent.types import PlannedAction
from jarvis_v2.memory.skill_projection import reconcile_skill_projection
from jarvis_v2.memory.store import MemoryStore, SkillRecord
from jarvis_v2.scripts.test_runtime import handle_runtime_case, make_temp_runtime


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


def assert_store_delete_skill_is_atomic() -> None:
    with TemporaryDirectory(prefix="jarvis-skill-delete-store-") as temp:
        store = MemoryStore(Path(temp) / "skills.sqlite")
        store.init()

        no_lookup_id = store.save_skill(SkillRecord("No Separate Lookup", "", "body"))
        original_get_skill = store.get_skill

        def reject_separate_lookup(name: str):
            raise AssertionError(f"delete_skill called get_skill for {name!r}")

        store.get_skill = reject_separate_lookup  # type: ignore[method-assign]
        try:
            if not store.delete_skill("No Separate Lookup"):
                raise SystemExit("delete_skill failed without a separate get_skill call")
        finally:
            store.get_skill = original_get_skill  # type: ignore[method-assign]
        with store.connect() as conn:
            if conn.execute("SELECT 1 FROM skills WHERE id = ?", (no_lookup_id,)).fetchone():
                raise SystemExit("delete_skill left the directly selected skill behind")

        ignored_id = store.save_skill(SkillRecord("Ignored Delete", "", "body"))
        with store.connect() as conn:
            conn.execute(
                """
                CREATE TRIGGER ignore_skill_delete
                BEFORE DELETE ON skills
                WHEN old.id = %d
                BEGIN
                    SELECT RAISE(IGNORE);
                END
                """ % ignored_id
            )
        if store.delete_skill("Ignored Delete"):
            raise SystemExit("delete_skill reported success for a zero-row DELETE")
        with store.connect() as conn:
            if conn.execute("SELECT 1 FROM skills WHERE id = ?", (ignored_id,)).fetchone() is None:
                raise SystemExit("zero-row delete regression did not preserve its target")
            conn.execute("DROP TRIGGER ignore_skill_delete")

        exact_id = store.save_skill(SkillRecord("Exact Target", "", "exact"))
        fuzzy_decoy_id = store.save_skill(SkillRecord("Newer Exact Target Copy", "", "decoy"))
        with store.connect() as conn:
            conn.execute("UPDATE skills SET updated_at = '2026-01-01T00:00:00Z' WHERE id = ?", (exact_id,))
            conn.execute("UPDATE skills SET updated_at = '2026-01-03T00:00:00Z' WHERE id = ?", (fuzzy_decoy_id,))
        if not store.delete_skill("Exact Target"):
            raise SystemExit("delete_skill exact match failed")
        with store.connect() as conn:
            if conn.execute("SELECT 1 FROM skills WHERE id = ?", (exact_id,)).fetchone() is not None:
                raise SystemExit("delete_skill did not prefer its exact match")
            if conn.execute("SELECT 1 FROM skills WHERE id = ?", (fuzzy_decoy_id,)).fetchone() is None:
                raise SystemExit("delete_skill exact match removed a newer fuzzy decoy")

        older_fuzzy_id = store.save_skill(SkillRecord("Older Fuzzy Needle", "", "older"))
        newer_fuzzy_id = store.save_skill(SkillRecord("Newer Fuzzy Needle", "", "newer"))
        with store.connect() as conn:
            conn.execute("UPDATE skills SET updated_at = '2026-01-01T00:00:00Z' WHERE id = ?", (older_fuzzy_id,))
            conn.execute("UPDATE skills SET updated_at = '2026-01-04T00:00:00Z' WHERE id = ?", (newer_fuzzy_id,))
        if not store.delete_skill("Fuzzy Needle"):
            raise SystemExit("delete_skill fuzzy match failed")
        with store.connect() as conn:
            if conn.execute("SELECT 1 FROM skills WHERE id = ?", (newer_fuzzy_id,)).fetchone() is not None:
                raise SystemExit("delete_skill did not choose the newest fuzzy match")
            if conn.execute("SELECT 1 FROM skills WHERE id = ?", (older_fuzzy_id,)).fetchone() is None:
                raise SystemExit("delete_skill removed the older fuzzy match")

        selected = threading.Event()
        release_delete = threading.Event()
        save_attempted = threading.Event()
        save_finished = threading.Event()

        class CoordinatedDeleteStore(MemoryStore):
            def connect(self):
                conn = super().connect()

                def trace(statement: str) -> None:
                    normalized = " ".join(statement.lower().split())
                    if normalized.startswith("select * from skills where lower(name) = lower("):
                        selected.set()
                        release_delete.wait(timeout=5.0)

                conn.set_trace_callback(trace)
                return conn

        class CoordinatedSaveStore(MemoryStore):
            def connect(self):
                conn = super().connect()

                def trace(statement: str) -> None:
                    normalized = " ".join(statement.lower().split())
                    if normalized.startswith("begin immediate"):
                        save_attempted.set()

                conn.set_trace_callback(trace)
                return conn

        concurrent_name = "Concurrent Skill"
        selected_id = store.save_skill(SkillRecord(concurrent_name, "", "before"))
        delete_store = CoordinatedDeleteStore(store.db_path)
        save_store = CoordinatedSaveStore(store.db_path)
        delete_result: list[bool] = []
        saved_ids: list[int] = []
        thread_errors: list[BaseException] = []

        def run_delete() -> None:
            try:
                delete_result.append(delete_store.delete_skill(concurrent_name))
            except BaseException as exc:
                thread_errors.append(exc)

        def run_save() -> None:
            try:
                saved_ids.append(save_store.save_skill(SkillRecord(concurrent_name, "", "after")))
            except BaseException as exc:
                thread_errors.append(exc)
            finally:
                save_finished.set()

        delete_thread = threading.Thread(target=run_delete, name="skill-delete-smoke")
        save_thread = threading.Thread(target=run_save, name="skill-save-smoke")
        delete_thread.start()
        try:
            if not selected.wait(timeout=5.0):
                raise SystemExit("concurrent delete did not reach its exact-match lookup")
            save_thread.start()
            if not save_attempted.wait(timeout=5.0):
                raise SystemExit("concurrent save did not reach SQLite")
            if save_finished.is_set():
                raise SystemExit("concurrent save was not serialized behind delete selection")
        finally:
            release_delete.set()
        delete_thread.join(timeout=5.0)
        save_thread.join(timeout=5.0)
        if delete_thread.is_alive() or save_thread.is_alive():
            raise SystemExit("concurrent skill save/delete did not finish")
        if thread_errors:
            raise SystemExit(f"concurrent skill save/delete failed: {thread_errors}")
        if delete_result != [True] or len(saved_ids) != 1:
            raise SystemExit(f"concurrent skill save/delete returned bad results: {delete_result} / {saved_ids}")
        if saved_ids[0] == selected_id:
            raise SystemExit("concurrent save updated the selected target before delete committed")
        with store.connect() as conn:
            if conn.execute("SELECT 1 FROM skills WHERE id = ?", (selected_id,)).fetchone() is not None:
                raise SystemExit("delete_skill reported success while its selected target survived")


def assert_same_timestamp_skill_order_is_newest_first() -> None:
    with TemporaryDirectory(prefix="jarvis-skill-order-") as temp:
        store = MemoryStore(Path(temp) / "skills.sqlite")
        store.init()
        older_id = store.save_skill(
            SkillRecord("Shared Ordering Skill Older", "shared ordering", "older body")
        )
        newer_id = store.save_skill(
            SkillRecord("Shared Ordering Skill Newer", "shared ordering", "newer body")
        )
        tied_time = "2026-08-14T00:00:00Z"
        with store.connect() as conn:
            conn.execute(
                "UPDATE skills SET updated_at = ? WHERE id IN (?, ?)",
                (tied_time, older_id, newer_id),
            )

        expected = [newer_id, older_id]
        result_sets = (
            store.list_skills(limit=2),
            store.list_active_skills(limit=2),
            store.search_skills("shared ordering", limit=2),
            store.search_active_skills("shared ordering", limit=2),
        )
        if any([int(row["id"]) for row in rows] != expected for rows in result_sets):
            raise SystemExit("same-timestamp skill ordering was not newest-first")
        fuzzy = store.get_skill("Ordering Skill")
        if fuzzy is None or int(fuzzy["id"]) != newer_id:
            raise SystemExit("same-timestamp fuzzy skill lookup was not newest-first")


def assert_skill_projection_identity_is_collision_safe() -> None:
    with TemporaryDirectory(prefix="jarvis-skill-projection-identity-") as temp:
        runtime = make_temp_runtime(Path(temp))
        save_skill = runtime.registry.get("save_skill").handler
        delete_tool = runtime.registry.get("delete_skill")

        def delete_skill(args: dict):
            resolver = delete_tool.approval_argument_resolver
            if resolver is None:
                raise SystemExit("delete_skill is missing its pre-approval target resolver")
            resolved = resolver(args)
            if hasattr(resolved, "args"):
                return runtime.executor.execute(
                    PlannedAction("delete_skill", resolved.args, "skill projection smoke"),
                    approved=True,
                )
            return resolved
        first = save_skill({"name": "Deploy/Audit", "trigger": "first", "body": "first body"})
        second = save_skill({"name": "Deploy:Audit", "trigger": "second", "body": "second body"})
        if not first.ok or not second.ok:
            raise SystemExit(f"colliding skill saves failed: {first} / {second}")
        first_path = Path(first.metadata["path"])
        second_path = Path(second.metadata["path"])
        if first_path == second_path or not first_path.exists() or not second_path.exists():
            raise SystemExit(f"colliding skills did not receive distinct notes: {first_path} / {second_path}")
        first_text = first_path.read_text(encoding="utf-8")
        second_text = second_path.read_text(encoding="utf-8")
        for result, text, name, body in (
            (first, first_text, "Deploy/Audit", "first body"),
            (second, second_text, "Deploy:Audit", "second body"),
        ):
            if (
                "jarvis_projection: skill" not in text
                or f"store_identity: {runtime.store.get_store_identity()}" not in text
                or f"skill_id: {result.metadata['skill_id']}" not in text
                or f"name: {name}" not in text
                or body not in text
            ):
                raise SystemExit(f"skill projection ownership/content diverged: {text}")

        updated = save_skill({"name": "Deploy/Audit", "trigger": "updated", "body": "updated first body"})
        if Path(updated.metadata["path"]) != first_path:
            raise SystemExit("same skill ID did not keep a stable projection path")
        if "updated first body" not in first_path.read_text(encoding="utf-8"):
            raise SystemExit("skill update did not refresh its owned projection")
        if second_path.read_text(encoding="utf-8") != second_text:
            raise SystemExit("skill update changed a colliding skill projection")

        deleted = delete_skill({"name": "Deploy/Audit"})
        if not deleted.ok or first_path.exists() or not second_path.exists():
            raise SystemExit(f"skill deletion crossed projection identities: {deleted}")
        if runtime.store.get_skill("Deploy:Audit") is None:
            raise SystemExit("skill deletion removed the colliding database row")

        prefix = "x" * 90
        long_first = save_skill({"name": prefix + " A", "body": "long first"})
        long_second = save_skill({"name": prefix + " B", "body": "long second"})
        if Path(long_first.metadata["path"]) == Path(long_second.metadata["path"]):
            raise SystemExit("90-character skill-name truncation lost the identity suffix")

        older = save_skill({"name": "Older Fuzzy Needle", "body": "older fuzzy"})
        newer = save_skill({"name": "Newer Fuzzy Needle", "body": "newer fuzzy"})
        with runtime.store.connect() as conn:
            conn.execute("UPDATE skills SET updated_at = '2026-07-11T00:00:00Z' WHERE id = ?", (older.metadata["skill_id"],))
            conn.execute("UPDATE skills SET updated_at = '2026-07-11T00:00:01Z' WHERE id = ?", (newer.metadata["skill_id"],))
        fuzzy_deleted = delete_skill({"name": "Fuzzy Needle"})
        if (
            fuzzy_deleted.ok
            or fuzzy_deleted.metadata.get("reason") != "ambiguous_target"
            or not Path(newer.metadata["path"]).exists()
            or not Path(older.metadata["path"]).exists()
            or runtime.store.get_skill("Older Fuzzy Needle") is None
            or runtime.store.get_skill("Newer Fuzzy Needle") is None
        ):
            raise SystemExit(f"ambiguous fuzzy delete did not fail closed: {fuzzy_deleted}")

    with TemporaryDirectory(prefix="jarvis-skill-projection-legacy-") as temp:
        runtime = make_temp_runtime(Path(temp))
        legacy_path = runtime.vault.root_path / "Skills" / "Legacy Skill.md"
        legacy_path.write_text(
            "---\nname: Legacy Skill\ntrigger: old trigger\ntags: old\n---\n\n"
            "# Legacy Skill\n\n## Trigger\n\nold trigger\n\n## Procedure\n\nold body\n",
            encoding="utf-8",
        )
        result = runtime.registry.get("save_skill").handler(
            {"name": "Legacy Skill", "trigger": "old trigger", "body": "old body", "tags": "old"}
        )
        if not result.ok or Path(result.metadata["path"]) != legacy_path:
            raise SystemExit(f"proven legacy skill projection was not reused: {result}")
        migrated = legacy_path.read_text(encoding="utf-8")
        if "jarvis_projection: skill" not in migrated or f"skill_id: {result.metadata['skill_id']}" not in migrated:
            raise SystemExit(f"legacy skill projection did not gain durable ownership: {migrated}")
        updated = runtime.registry.get("save_skill").handler(
            {"name": "Legacy Skill", "trigger": "new trigger", "body": "new body", "tags": "new"}
        )
        if not updated.ok or Path(updated.metadata["path"]) != legacy_path:
            raise SystemExit(f"owned legacy skill projection was not reused: {updated}")
        migrated = legacy_path.read_text(encoding="utf-8")
        if "new body" not in migrated or "skill_revision: 2" not in migrated:
            raise SystemExit(f"owned legacy skill projection did not update safely: {migrated}")

        other_store = MemoryStore(Path(temp) / "other.sqlite")
        other_store.init()
        other_skill_id = other_store.save_skill(
            SkillRecord("Legacy Skill", "other trigger", "other body", "other")
        )
        other_projection = reconcile_skill_projection(
            other_store, runtime.vault, other_skill_id
        )
        if other_projection.status != "completed":
            raise SystemExit(f"second store projection did not complete: {other_projection}")
        other_path = runtime.vault.root_path / other_projection.path_display
        if legacy_path.read_text(encoding="utf-8") != migrated:
            raise SystemExit("a second store overwrote another store's owned skill projection")
        if other_path == legacy_path or not other_path.exists():
            raise SystemExit("a second store did not receive a separate owned skill projection")
        other_text = other_path.read_text(encoding="utf-8")
        if (
            f"store_identity: {other_store.get_store_identity()}" not in other_text
            or f"skill_id: {other_skill_id}" not in other_text
            or "other body" not in other_text
        ):
            raise SystemExit(f"second-store skill projection lost its ownership/content: {other_text}")
            replacement = conn.execute("SELECT id, body FROM skills WHERE name = ?", (concurrent_name,)).fetchone()
            if replacement is None or dict(replacement) != {"id": saved_ids[0], "body": "after"}:
                raise SystemExit(f"serialized concurrent save was not preserved: {replacement}")


def assert_route(command: str, tool_name: str, args: dict | None = None) -> None:
    plan = RuleBasedPlanner().plan(command)
    actions = plan.actions
    if len(actions) != 1 or actions[0].tool_name != tool_name:
        raise SystemExit(f"{command!r} should route to {tool_name}: {[(a.tool_name, a.args) for a in actions]}")
    expected_args = args or {}
    if actions[0].args != expected_args:
        raise SystemExit(f"{command!r} should pass {expected_args}: {actions[0].args}")


def test_skill_read_alias_routes() -> None:
    for command in (
        "skills please",
        "skill list please",
        "show latest skills",
        "saved skills please",
        "show saved skills",
        # Real gap found live 2026-07-09: "show my skills" fell through to
        # chat while bare "list skills" worked.
        "show my skills",
        # Real gap found live 2026-07-10, same class: "show my skills" worked
        # but bare "my skills"/"list my skills" and any "recent" qualifier
        # fell through to chat.
        "my skills",
        "list my skills",
        "recent skills",
        "show my recent skills",
        "list my recent skills",
        "my recent skills",
        # Real gap found live 2026-07-10 (round 43): trailing "please" broke
        # this whole exact-match set, same class as the notes fix -- see
        # smoke_test_notes.py's matching comment for the full root cause.
        # Fixed with a local trailing-only strip.
        "show my recent skills please",
    ):
        assert_route(command, "list_skills")
    assert_route("search skills for memory please", "search_skills", {"query": "memory"})


def assert_runtime_routes_skill_count_aliases() -> None:
    with TemporaryDirectory(prefix="jarvis-skill-count-alias-") as temp:
        runtime = make_temp_runtime(Path(temp))
        for text in (
            "skill count",
            "skills count",
            "count skills",
            "how many skills do i have",
            "스킬 몇 개",
            "스킬 개수",
        ):
            result = runtime.handle(text)
            planned = (result.metadata or {}).get("runtime_trace", {}).get("planned_actions") or []
            if [action.get("tool_name") for action in planned] != ["list_skills"]:
                raise SystemExit(f"runtime missed skill-count alias {text!r}: {planned} / {result.response!r}")
            if len(result.tool_results) != 1 or result.tool_results[0].tool_name != "list_skills":
                raise SystemExit(f"skill-count alias should execute one list_skills tool for {text!r}: {result.tool_results}")
            metadata = result.tool_results[0].metadata
            if "Count: 0" not in result.response or metadata.get("total_skills") != 0:
                raise SystemExit(f"empty skill-count alias should answer with zero count for {text!r}: {result.response!r} / {metadata}")
            for key in (
                "writes_files",
                "writes_memory",
                "writes_notes",
                "writes_skills",
                "queues_approval",
                "controls_computer",
                "reads_private_data",
                "authorizes_execution",
                "authorizes_completion_claim",
                "approval_granted",
            ):
                if metadata.get(key):
                    raise SystemExit(f"skill-count alias unexpectedly set {key} for {text!r}: {metadata}")
        if runtime.store.list_skills(limit=100):
            raise SystemExit("skill count aliases must not create or mutate skills.")
        if runtime.store.list_memories(limit=100):
            raise SystemExit("skill count aliases must not create memory rows.")


def assert_linked_skill_review_token(metadata: dict, label: str) -> None:
    token = str(metadata.get("linked_skill_review_token_sha256") or "")
    rows = metadata.get("linked_skill_review_rows") or []
    if metadata.get("linked_skill_review_token_present") is not True or len(token) != 64:
        raise SystemExit(f"{label} missed linked skill review token: {metadata}")
    if metadata.get("linked_skill_review_row_count") != len(rows) or len(rows) < 5:
        raise SystemExit(f"{label} missed linked skill review rows: {metadata}")
    for flag in [
        "linked_skill_review_token_authorizes_install",
        "linked_skill_review_token_authorizes_tool_execution",
        "linked_skill_review_token_authorizes_approval",
        "linked_skill_review_token_authorizes_risky_action",
        "linked_skill_review_token_reusable_for_next_link",
    ]:
        if metadata.get(flag) is not False:
            raise SystemExit(f"{label} should keep {flag}=False: {metadata}")
    if metadata.get("next_linked_skill_intake_requires_fresh_review_token") is not True:
        raise SystemExit(f"{label} missed fresh linked skill intake token requirement: {metadata}")
    for row in rows:
        for flag in [
            "authorizes_install",
            "authorizes_tool_execution",
            "authorizes_approval",
            "authorizes_risky_action",
            "reusable_for_next_link",
        ]:
            if row.get(flag) is not False:
                raise SystemExit(f"{label} row should keep {flag}=False: {metadata}")


def assert_skill_write_receipt(result, *, root: Path, label: str) -> None:
    metadata = result.metadata
    path = metadata.get("path")
    path_display = metadata.get("path_display")
    if not isinstance(path, str) or not Path(path).exists():
        raise SystemExit(f"{label} did not preserve an exact saved skill path: {metadata}")
    if not isinstance(path_display, str) or not path_display.startswith("Skills/"):
        raise SystemExit(f"{label} missed safe Skills path_display: {metadata}")
    if f"Saved note: {path_display}" not in result.output:
        raise SystemExit(f"{label} missed safe saved-note receipt: {result.output}")
    if str(root) in result.output or "/var/folders/" in result.output or "/private/" in result.output or "/\x55sers/" in result.output:
        raise SystemExit(f"{label} leaked a local path in output: {result.output}")


def assert_linked_skill_write_receipts(result, *, root: Path, label: str) -> None:
    metadata = result.metadata
    installed = metadata.get("skills") or []
    if not installed:
        raise SystemExit(f"{label} missed installed skill metadata: {metadata}")
    for item in installed:
        path = item.get("path")
        path_display = item.get("path_display")
        if not isinstance(path, str) or not Path(path).exists():
            raise SystemExit(f"{label} missed exact installed path: {metadata}")
        if not isinstance(path_display, str) or not path_display.startswith("Skills/"):
            raise SystemExit(f"{label} missed safe installed path_display: {metadata}")
        if path_display not in result.output:
            raise SystemExit(f"{label} output missed installed display path {path_display!r}: {result.output}")
    if str(root) in result.output or "/var/folders/" in result.output or "/private/" in result.output or "/\x55sers/" in result.output:
        raise SystemExit(f"{label} leaked a local path in output: {result.output}")


def assert_no_local_path(value: str, label: str) -> None:
    for fragment in ["/\x55sers/", "/private/", "/var/folders/", "/tmp/"]:
        if fragment in value:
            raise SystemExit(f"{label} leaked a local path: {value}")


def assert_skill_refusal_handoff(
    metadata: dict,
    label: str,
    *,
    source: str,
    mutation: str,
    reason: str,
    read_only: bool,
    requires_approval: bool = False,
) -> None:
    handoff = metadata.get("skill_refusal_handoff")
    if metadata.get("skill_refusal_handoff_ready") is not True or not isinstance(handoff, dict):
        raise SystemExit(f"{label} should emit a ready skill_refusal_handoff: {metadata}")
    if handoff.get("source") != source or handoff.get("ready_for_operator") is not True:
        raise SystemExit(f"{label} has wrong skill refusal source/readiness: {handoff}")
    if handoff.get("mutation") != mutation or handoff.get("reason") != reason or handoff.get("refused") is not True:
        raise SystemExit(f"{label} has wrong skill refusal mutation/reason: {handoff}")
    if handoff.get("changed") != []:
        raise SystemExit(f"{label} should not report changed skill rows: {handoff}")
    if not isinstance(handoff.get("next_commands"), dict) or "list skills" not in handoff["next_commands"].values():
        raise SystemExit(f"{label} should include skill recovery commands: {handoff}")
    for key in ("skill_id", "raw_name", "raw_query", "query_chars", "chars", "max_chars"):
        if key in handoff and key in metadata and handoff.get(key) != metadata.get(key):
            raise SystemExit(f"{label} handoff {key} diverged from metadata: {handoff} vs {metadata}")
    boundaries = handoff.get("boundaries") or {}
    expected = {
        "read_only": read_only,
        "writes_files": False,
        "writes_memory": False,
        "writes_notes": False,
        "writes_skills": False,
        "writes_database": False,
        "queues_approval": False,
        "requires_approval": requires_approval,
        "controls_computer": False,
        "external_side_effect": False,
        "calls_model": False,
        "executes_tools": False,
        "authorizes_execution": False,
        "authorizes_completion_claim": False,
        "approval_granted": False,
        "completes_tasks": False,
        "creates_skill": False,
        "deletes_skill": False,
        "reads_private_data": False,
    }
    for key, value in expected.items():
        if boundaries.get(key) is not value:
            raise SystemExit(f"{label} refusal boundary {key} should be {value}: {boundaries}")
    assert_no_local_path(repr(handoff), label)


def main() -> None:
    assert_store_delete_skill_is_atomic()
    assert_same_timestamp_skill_order_is_newest_first()
    assert_skill_projection_identity_is_collision_safe()
    test_skill_read_alias_routes()
    assert_runtime_routes_skill_count_aliases()
    with TemporaryDirectory(prefix="jarvis-skills-") as temp:
        runtime = make_temp_runtime(Path(temp))
        root = runtime.vault.root_path
        cases = [
            "extract linked skills from https://github.com/multica-ai/andrej-karpathy-skills.git https://github.com/NousResearch/hermes-agent https://github.com/tinyhumansai/openhuman",
            "install linked skills from https://github.com/multica-ai/andrej-karpathy-skills.git https://github.com/NousResearch/hermes-agent https://github.com/tinyhumansai/openhuman",
            "skill match preview: import documents and notes into Jarvis memory",
            "search skills OpenHuman",
            "get skill OpenHuman Memory Tree Ingest",
            "search skills Karpathy",
            "get skill Karpathy Goal-Driven Execution",
            "help skills",
            (
                "save skill V2 Smoke Test Skill when testing Jarvis skill storage do "
                "Run the skill smoke test, confirm SQLite stores it, and confirm Obsidian writes it."
            ),
            "search skills smoke test",
            "get skill V2 Smoke Test Skill",
            "list skills",
            "delete skill V2 Smoke Test Skill",
        ]
        for case in cases:
            result = handle_runtime_case(runtime, case, approved=case.startswith("delete skill"))
            status = "ok" if result.verified else "failed"
            print(f"[{status}] {case}")
            print(result.response)
            print()
            if case.startswith("extract linked skills"):
                required = [
                    "Linked skill extraction",
                    "Karpathy Goal-Driven Execution",
                    "Hermes Learning Loop",
                    "OpenHuman Memory Tree Ingest",
                    "Linked skill review boundary",
                    "review token authorizes risky action: no",
                    "read-only",
                    "without fetching pages",
                    "Install path",
                ]
                missing = [item for item in required if item not in result.response]
                if missing:
                    raise SystemExit(f"Linked skill extraction missing expected text: {missing}")
                metadata = result.tool_results[0].metadata
                if metadata.get("writes_skills") or metadata.get("candidates", 0) < 4:
                    raise SystemExit("Linked skill extraction should be read-only and produce multiple candidates.")
                if metadata.get("writes_files") is not False or metadata.get("reads_private_data") is not False:
                    raise SystemExit("Linked skill extraction missed safety metadata.")
                assert_linked_skill_review_token(metadata, "extract linked skills")
            if case.startswith("install linked skills"):
                required = [
                    "Installed linked skills",
                    "Karpathy Goal-Driven Execution",
                    "Hermes Learning Loop",
                    "OpenHuman Memory Tree Ingest",
                    "Safety boundary",
                    "did not fetch links",
                    "linked skill review token sha256",
                ]
                missing = [item for item in required if item not in result.response]
                if missing:
                    raise SystemExit(f"Linked skill install missing expected text: {missing}")
                metadata = result.tool_results[0].metadata
                if not metadata.get("writes_skills") or metadata.get("installed", 0) < 4:
                    raise SystemExit("Linked skill install should save multiple local skills.")
                if metadata.get("limit") != 6 or metadata.get("writes_files") is not True:
                    raise SystemExit("Linked skill install missed limit/write metadata.")
                if metadata.get("writes_memory") is not True or metadata.get("writes_notes") is not True:
                    raise SystemExit("Linked skill install missed memory/note write metadata.")
                assert_linked_skill_review_token(metadata, "install linked skills")
                assert_linked_skill_write_receipts(result.tool_results[0], root=root, label="runtime install_linked_skills")
            if case.startswith("skill match preview"):
                required = [
                    "Jarvis skill match preview",
                    "read-only",
                    "Request",
                    "Candidate skills",
                    "OpenHuman Memory Tree Ingest",
                    "Use boundary",
                    "does not bypass planner",
                ]
                missing = [item for item in required if item not in result.response]
                if missing:
                    raise SystemExit(f"Skill match preview missing expected text: {missing}")
                metadata = result.tool_results[0].metadata
                if (
                    metadata.get("executes_tools")
                    or metadata.get("writes_memory")
                    or metadata.get("writes_skills")
                    or metadata.get("queues_approval")
                ):
                    raise SystemExit("Skill match preview must not execute tools, write memory/skills, or queue approvals.")
            if case == "search skills OpenHuman" and "OpenHuman Memory Tree Ingest" not in result.response:
                raise SystemExit("OpenHuman linked skill should be searchable after install.")
            if case == "search skills OpenHuman":
                metadata = result.tool_results[0].metadata
                if metadata.get("limit") != 20 or metadata.get("writes_skills") is not False:
                    raise SystemExit("Search skills missed sanitized limit/read-only metadata.")
            if case == "get skill OpenHuman Memory Tree Ingest":
                required = ["OpenHuman Memory Tree Ingest", "Source inspiration", "Obsidian", "SQLite"]
                missing = [item for item in required if item not in result.response]
                if missing:
                    raise SystemExit(f"OpenHuman linked skill readback missing expected text: {missing}")
                metadata = result.tool_results[0].metadata
                if metadata.get("skill_id") is None or metadata.get("writes_files") is not False:
                    raise SystemExit("Get skill missed read-only metadata.")
            if case == "search skills Karpathy" and "Karpathy Goal-Driven Execution" not in result.response:
                raise SystemExit("Karpathy linked skill should be searchable after install.")
            if case == "get skill Karpathy Goal-Driven Execution":
                required = ["Karpathy Goal-Driven Execution", "Source inspiration", "Surface assumptions", "focused verification"]
                missing = [item for item in required if item not in result.response]
                if missing:
                    raise SystemExit(f"Karpathy linked skill readback missing expected text: {missing}")
                metadata = result.tool_results[0].metadata
                if metadata.get("skill_id") is None or metadata.get("writes_files") is not False:
                    raise SystemExit("Get Karpathy skill missed read-only metadata.")
            if case == "help skills":
                required = [
                    "skill match preview",
                    "extract linked skills",
                    "andrej-karpathy-skills",
                    "hermes-agent",
                    "openhuman",
                    "install linked skills",
                    "save skill",
                ]
                missing = [item for item in required if item not in result.response]
                if missing:
                    raise SystemExit(f"Help skills missing expected text: {missing}")
            if case.startswith("save skill V2 Smoke Test Skill"):
                metadata = result.tool_results[0].metadata
                if metadata.get("writes_skills") is not True or metadata.get("writes_files") is not True:
                    raise SystemExit("Save skill missed write metadata.")
                if metadata.get("writes_memory") is not True or metadata.get("writes_notes") is not True:
                    raise SystemExit("Save skill missed memory/note write metadata.")
                assert_skill_write_receipt(result.tool_results[0], root=root, label="runtime save_skill")
            if case == "list skills":
                metadata = result.tool_results[0].metadata
                if metadata.get("limit") != 25 or metadata.get("writes_skills") is not False:
                    raise SystemExit("List skills missed sanitized limit/read-only metadata.")
                if "Count:" not in result.response or metadata.get("total_skills") != metadata.get("count"):
                    raise SystemExit(f"list_skills missed visible/metadata count parity: {result.response!r} / {metadata}")
            if case == "delete skill V2 Smoke Test Skill":
                metadata = result.tool_results[0].metadata
                if metadata.get("writes_skills") is not True:
                    raise SystemExit("Delete skill missed delete metadata.")

        direct_list = runtime.registry.get("list_skills").handler({"limit": "bad"})
        if not direct_list.ok or direct_list.metadata.get("limit") != 25 or direct_list.metadata.get("raw_limit") != "bad":
            raise SystemExit("list_skills did not sanitize a bad limit.")
        direct_list_bool = runtime.registry.get("list_skills").handler({"limit": False})
        if not direct_list_bool.ok or direct_list_bool.metadata.get("limit") != 25 or direct_list_bool.metadata.get("raw_limit") != "False":
            raise SystemExit(f"list_skills should treat boolean limits as malformed defaults: {direct_list_bool.metadata}")
        if direct_list_bool.metadata.get("writes_skills") or direct_list_bool.metadata.get("writes_files"):
            raise SystemExit(f"list_skills boolean limit should stay read-only: {direct_list_bool.metadata}")
        direct_list_long_limit = runtime.registry.get("list_skills").handler({"limit": "l" * 200})
        if direct_list_long_limit.metadata.get("raw_limit") != ("l" * 77 + "..."):
            raise SystemExit(f"list_skills did not bound raw bad limit metadata: {direct_list_long_limit.metadata}")
        path_bad_list_limit = runtime.registry.get("list_skills").handler({"limit": "/\x55sers/example/private/skill-limit"})
        if path_bad_list_limit.metadata.get("raw_limit") != "<local-path>":
            raise SystemExit(f"list_skills should redact path-shaped bad limits: {path_bad_list_limit.metadata}")
        for value in ["/var/folders/zc/jarvis-skill-limit", "/tmp/jarvis-skill-limit"]:
            path_bad_temp_list_limit = runtime.registry.get("list_skills").handler({"limit": value})
            if path_bad_temp_list_limit.metadata.get("raw_limit") != "<local-path>":
                raise SystemExit(f"list_skills should redact temp-root bad limits: {path_bad_temp_list_limit.metadata}")

        malformed_runtime = make_temp_runtime(Path(temp) / "malformed")
        leak_markers = [
            "SKILL_LIST_ROW_SECRET",
            "SKILL_SEARCH_ROW_SECRET",
            "SKILL_READ_ROW_SECRET",
            "SKILL_PREVIEW_ROW_SECRET",
        ]
        malformed_runtime.store.list_skills = lambda limit=25: [
            HostileRow(leak_markers[0]),
            {"id": 41, "name": "Safe List Skill", "trigger": "when testing", "body": "Use guarded rows.", "tags": "safety"},
        ][:limit]
        malformed_list = malformed_runtime.registry.get("list_skills").handler({"limit": 10})
        malformed_list_text = malformed_list.output + repr(malformed_list.metadata)
        if not malformed_list.ok:
            raise SystemExit(f"list_skills should survive hostile local rows: {malformed_list.output}")
        if any(marker in malformed_list_text for marker in leak_markers):
            raise SystemExit(f"list_skills leaked hostile row text: {malformed_list_text}")
        if "Safe List Skill" not in malformed_list.output or "could not be read safely" not in malformed_list.output:
            raise SystemExit(f"list_skills should preserve readable rows and report unreadable rows: {malformed_list.output}")
        if malformed_list.metadata.get("count") != 2 or malformed_list.metadata.get("readable_skill_rows") != 1 or malformed_list.metadata.get("unreadable_skill_rows") != 1:
            raise SystemExit(f"list_skills hostile row case missed readable/unreadable metadata: {malformed_list.metadata}")

        malformed_runtime.store.search_skills = lambda query, limit=20: [
            HostileRow(leak_markers[1]),
            {"id": 42, "name": "Safe Search Skill", "trigger": "when searching", "body": "Search safely.", "tags": "search"},
        ][:limit]
        malformed_search = malformed_runtime.registry.get("search_skills").handler({"query": "safe"})
        malformed_search_text = malformed_search.output + repr(malformed_search.metadata)
        if not malformed_search.ok:
            raise SystemExit(f"search_skills should survive hostile local rows: {malformed_search.output}")
        if any(marker in malformed_search_text for marker in leak_markers):
            raise SystemExit(f"search_skills leaked hostile row text: {malformed_search_text}")
        if "Safe Search Skill" not in malformed_search.output or "could not be read safely" not in malformed_search.output:
            raise SystemExit(f"search_skills should preserve readable rows and report unreadable rows: {malformed_search.output}")
        if malformed_search.metadata.get("count") != 2 or malformed_search.metadata.get("readable_skill_rows") != 1 or malformed_search.metadata.get("unreadable_skill_rows") != 1:
            raise SystemExit(f"search_skills hostile row case missed readable/unreadable metadata: {malformed_search.metadata}")

        malformed_runtime.store.get_skill = lambda name: HostileRow(leak_markers[2])
        malformed_get = malformed_runtime.registry.get("get_skill").handler({"name": "Safe Search Skill"})
        malformed_get_text = malformed_get.output + repr(malformed_get.metadata)
        if not malformed_get.ok:
            raise SystemExit(f"get_skill should fail closed without crashing on hostile rows: {malformed_get.output}")
        if any(marker in malformed_get_text for marker in leak_markers):
            raise SystemExit(f"get_skill leaked hostile row text: {malformed_get_text}")
        if "could not be read safely" not in malformed_get.output:
            raise SystemExit(f"get_skill hostile row case should explain safe unreadable state: {malformed_get.output}")
        if malformed_get.metadata.get("skill_status") != "unreadable" or malformed_get.metadata.get("skill_id") is not None:
            raise SystemExit(f"get_skill hostile row case missed unreadable metadata: {malformed_get.metadata}")
        if malformed_get.metadata.get("readable_skill_row") is not False or malformed_get.metadata.get("unreadable_skill_row") is not True:
            raise SystemExit(f"get_skill hostile row case missed readable/unreadable flags: {malformed_get.metadata}")

        malformed_runtime.store.search_active_skills = lambda query, limit=20: [
            HostileRow(leak_markers[3]),
            {"id": 43, "name": "Safe Preview Skill", "trigger": "when previewing", "body": "Preview safely.", "tags": "preview", "review_status": "active"},
        ][:limit]
        malformed_preview = malformed_runtime.registry.get("skill_match_preview").handler({"request": "preview safe behavior"})
        malformed_preview_text = malformed_preview.output + repr(malformed_preview.metadata)
        if not malformed_preview.ok:
            raise SystemExit(f"skill_match_preview should survive hostile local rows: {malformed_preview.output}")
        if any(marker in malformed_preview_text for marker in leak_markers):
            raise SystemExit(f"skill_match_preview leaked hostile row text: {malformed_preview_text}")
        if "Safe Preview Skill" not in malformed_preview.output:
            raise SystemExit(f"skill_match_preview should preserve readable candidate rows: {malformed_preview.output}")
        if malformed_preview.metadata.get("match_names") != ["Safe Preview Skill"]:
            raise SystemExit(f"skill_match_preview hostile row case missed safe match names: {malformed_preview.metadata}")

        direct_search = runtime.registry.get("search_skills").handler({"query": "OpenHuman", "limit": 999999})
        if not direct_search.ok or direct_search.metadata.get("limit") != 200:
            raise SystemExit("search_skills did not clamp a large limit.")
        direct_search_bool = runtime.registry.get("search_skills").handler({"query": "OpenHuman", "limit": True})
        if not direct_search_bool.ok or direct_search_bool.metadata.get("limit") != 20 or direct_search_bool.metadata.get("raw_limit") != "True":
            raise SystemExit(f"search_skills should treat boolean limits as malformed defaults: {direct_search_bool.metadata}")
        if direct_search_bool.metadata.get("writes_skills") or direct_search_bool.metadata.get("writes_files"):
            raise SystemExit(f"search_skills boolean limit should stay read-only: {direct_search_bool.metadata}")
        path_bad_search_limit = runtime.registry.get("search_skills").handler({"query": "OpenHuman", "limit": "/private/tmp/jarvis-skill-search-limit"})
        if not path_bad_search_limit.ok or path_bad_search_limit.metadata.get("raw_limit") != "<local-path>":
            raise SystemExit(f"search_skills should redact path-shaped bad limits: {path_bad_search_limit.metadata}")
        for value in ["/var/folders/zc/jarvis-skill-search-limit", "/tmp/jarvis-skill-search-limit"]:
            path_bad_temp_search_limit = runtime.registry.get("search_skills").handler({"query": "OpenHuman", "limit": value})
            if not path_bad_temp_search_limit.ok or path_bad_temp_search_limit.metadata.get("raw_limit") != "<local-path>":
                raise SystemExit(f"search_skills should redact temp-root bad limits: {path_bad_temp_search_limit.metadata}")

        direct_install = runtime.registry.get("install_linked_skills").handler({"source": "openhuman hermes-agent", "limit": "bad"})
        if not direct_install.ok or direct_install.metadata.get("limit") != 6 or direct_install.metadata.get("raw_limit") != "bad":
            raise SystemExit("install_linked_skills did not sanitize a bad limit.")
        assert_linked_skill_write_receipts(direct_install, root=root, label="direct install_linked_skills")
        direct_install_bool = runtime.registry.get("install_linked_skills").handler({"source": "openhuman hermes-agent", "limit": False})
        if not direct_install_bool.ok or direct_install_bool.metadata.get("limit") != 6 or direct_install_bool.metadata.get("raw_limit") != "False":
            raise SystemExit(f"install_linked_skills should treat boolean limits as malformed defaults: {direct_install_bool.metadata}")
        if direct_install_bool.metadata.get("installed", 0) < 2 or direct_install_bool.metadata.get("writes_skills") is not True:
            raise SystemExit(f"install_linked_skills boolean limit should preserve normal local install behavior: {direct_install_bool.metadata}")
        assert_linked_skill_write_receipts(direct_install_bool, root=root, label="boolean-limit install_linked_skills")
        path_bad_install_limit = runtime.registry.get("install_linked_skills").handler({"source": "openhuman hermes-agent", "limit": "/\x55sers/example/private/install-skill-limit"})
        if not path_bad_install_limit.ok or path_bad_install_limit.metadata.get("raw_limit") != "<local-path>":
            raise SystemExit(f"install_linked_skills should redact path-shaped bad limits: {path_bad_install_limit.metadata}")
        assert_linked_skill_write_receipts(path_bad_install_limit, root=root, label="path-limit install_linked_skills")
        for value in ["/var/folders/zc/install-skill-limit", "/tmp/install-skill-limit"]:
            path_bad_temp_install_limit = runtime.registry.get("install_linked_skills").handler({"source": "openhuman hermes-agent", "limit": value})
            if not path_bad_temp_install_limit.ok or path_bad_temp_install_limit.metadata.get("raw_limit") != "<local-path>":
                raise SystemExit(f"install_linked_skills should redact temp-root bad limits: {path_bad_temp_install_limit.metadata}")
            assert_linked_skill_write_receipts(path_bad_temp_install_limit, root=root, label="temp-path-limit install_linked_skills")

        oversized = runtime.registry.get("save_skill").handler({"name": "Oversized", "body": "x" * 50001})
        if oversized.ok or "Refusing to save" not in oversized.output:
            raise SystemExit("save_skill did not refuse oversized skill bodies.")
        if oversized.metadata.get("max_chars") != 50000 or oversized.metadata.get("writes_skills") is not False:
            raise SystemExit("save_skill oversized refusal missed metadata.")
        assert_skill_refusal_handoff(
            oversized.metadata,
            "save_skill oversized body",
            source="save_skill",
            mutation="skill_create",
            reason="body_too_large",
            read_only=False,
        )

        missing_name = runtime.registry.get("save_skill").handler({"name": "", "body": "do useful work"})
        if missing_name.ok or missing_name.metadata.get("reason") != "missing_name":
            raise SystemExit("save_skill missing name missed safe refusal metadata.")
        if missing_name.metadata.get("raw_name") != "":
            raise SystemExit(f"save_skill missing name missed bounded raw name metadata: {missing_name.metadata}")
        assert_skill_refusal_handoff(
            missing_name.metadata,
            "save_skill missing name",
            source="save_skill",
            mutation="skill_create",
            reason="missing_name",
            read_only=False,
        )
        path_bad_save_name = runtime.registry.get("save_skill").handler({"name": "/\x55sers/example/private/skill-name", "body": "do useful work"})
        if path_bad_save_name.ok or path_bad_save_name.metadata.get("reason") != "invalid_name":
            raise SystemExit("save_skill path-shaped name missed safe refusal metadata.")
        if path_bad_save_name.metadata.get("raw_name") != "<local-path>":
            raise SystemExit(f"save_skill leaked local path in raw name metadata: {path_bad_save_name.metadata}")
        if path_bad_save_name.metadata.get("writes_files") or path_bad_save_name.metadata.get("writes_memory") or path_bad_save_name.metadata.get("writes_notes") or path_bad_save_name.metadata.get("writes_skills"):
            raise SystemExit(f"save_skill path-shaped name should not write: {path_bad_save_name.metadata}")
        if "/\x55sers/" in path_bad_save_name.output or "/private/" in path_bad_save_name.output:
            raise SystemExit(f"save_skill leaked local path in refusal output: {path_bad_save_name.output}")
        assert_skill_refusal_handoff(
            path_bad_save_name.metadata,
            "save_skill path-shaped name",
            source="save_skill",
            mutation="skill_create",
            reason="invalid_name",
            read_only=False,
        )
        for value in ["/var/folders/zc/skill-name", "/tmp/skill-name"]:
            path_bad_temp_save_name = runtime.registry.get("save_skill").handler({"name": value, "body": "do useful work"})
            if path_bad_temp_save_name.ok or path_bad_temp_save_name.metadata.get("reason") != "invalid_name":
                raise SystemExit(f"save_skill temp-root name missed safe refusal metadata: {path_bad_temp_save_name.metadata}")
            if path_bad_temp_save_name.metadata.get("raw_name") != "<local-path>":
                raise SystemExit(f"save_skill leaked temp-root path in raw name metadata: {path_bad_temp_save_name.metadata}")
            assert_no_local_path(path_bad_temp_save_name.output, "save_skill temp-root refusal output")

        missing_query = runtime.registry.get("search_skills").handler({"query": "", "limit": "bad"})
        if missing_query.ok or missing_query.metadata.get("reason") != "missing_query":
            raise SystemExit("search_skills missing query missed safe refusal metadata.")
        if missing_query.metadata.get("query_chars") != 0 or missing_query.metadata.get("limit") != 20 or missing_query.metadata.get("raw_limit") != "bad":
            raise SystemExit(f"search_skills missing query missed bounded raw limit metadata: {missing_query.metadata}")
        assert_skill_refusal_handoff(
            missing_query.metadata,
            "search_skills missing query",
            source="search_skills",
            mutation="skill_search",
            reason="missing_query",
            read_only=True,
        )
        missing_query_path_limit = runtime.registry.get("search_skills").handler({"query": "", "limit": "/private/tmp/jarvis-missing-skill-query-limit"})
        if missing_query_path_limit.ok or missing_query_path_limit.metadata.get("raw_limit") != "<local-path>":
            raise SystemExit(f"search_skills missing query should redact path-shaped bad limits: {missing_query_path_limit.metadata}")
        for value in ["/var/folders/zc/jarvis-missing-skill-query-limit", "/tmp/jarvis-missing-skill-query-limit"]:
            missing_query_temp_limit = runtime.registry.get("search_skills").handler({"query": "", "limit": value})
            if missing_query_temp_limit.ok or missing_query_temp_limit.metadata.get("raw_limit") != "<local-path>":
                raise SystemExit(f"search_skills missing query should redact temp-root bad limits: {missing_query_temp_limit.metadata}")
        path_bad_search_query = runtime.registry.get("search_skills").handler({"query": "/\x55sers/example/private/skill-query"})
        if path_bad_search_query.ok or path_bad_search_query.metadata.get("reason") != "invalid_query":
            raise SystemExit("search_skills path-shaped query missed safe refusal metadata.")
        if path_bad_search_query.metadata.get("raw_query") != "<local-path>":
            raise SystemExit(f"search_skills leaked local path in raw query metadata: {path_bad_search_query.metadata}")
        if path_bad_search_query.metadata.get("writes_files") or path_bad_search_query.metadata.get("writes_skills"):
            raise SystemExit(f"search_skills path-shaped query should stay read-only: {path_bad_search_query.metadata}")
        if "/\x55sers/" in path_bad_search_query.output or "/private/" in path_bad_search_query.output:
            raise SystemExit(f"search_skills leaked local path in refusal output: {path_bad_search_query.output}")
        assert_skill_refusal_handoff(
            path_bad_search_query.metadata,
            "search_skills path-shaped query",
            source="search_skills",
            mutation="skill_search",
            reason="invalid_query",
            read_only=True,
        )
        for value in ["/var/folders/zc/skill-query", "/tmp/skill-query"]:
            path_bad_temp_search_query = runtime.registry.get("search_skills").handler({"query": value})
            if path_bad_temp_search_query.ok or path_bad_temp_search_query.metadata.get("reason") != "invalid_query":
                raise SystemExit(f"search_skills temp-root query missed safe refusal metadata: {path_bad_temp_search_query.metadata}")
            if path_bad_temp_search_query.metadata.get("raw_query") != "<local-path>":
                raise SystemExit(f"search_skills leaked temp-root path in raw query metadata: {path_bad_temp_search_query.metadata}")
            assert_no_local_path(path_bad_temp_search_query.output, "search_skills temp-root refusal output")

        missing_get = runtime.registry.get("get_skill").handler({"name": ""})
        if missing_get.ok or missing_get.metadata.get("raw_name") != "":
            raise SystemExit(f"get_skill missing name missed bounded raw name metadata: {missing_get.metadata}")
        assert_skill_refusal_handoff(
            missing_get.metadata,
            "get_skill missing name",
            source="get_skill",
            mutation="skill_read",
            reason="missing_name",
            read_only=True,
        )
        path_bad_get_name = runtime.registry.get("get_skill").handler({"name": "/private/tmp/jarvis-skill-name"})
        if path_bad_get_name.ok or path_bad_get_name.metadata.get("reason") != "invalid_name":
            raise SystemExit("get_skill path-shaped name missed safe refusal metadata.")
        if path_bad_get_name.metadata.get("raw_name") != "<local-path>":
            raise SystemExit(f"get_skill leaked local path in raw name metadata: {path_bad_get_name.metadata}")
        if path_bad_get_name.metadata.get("writes_files") or path_bad_get_name.metadata.get("writes_skills"):
            raise SystemExit(f"get_skill path-shaped name should stay read-only: {path_bad_get_name.metadata}")
        if "/\x55sers/" in path_bad_get_name.output or "/private/" in path_bad_get_name.output:
            raise SystemExit(f"get_skill leaked local path in refusal output: {path_bad_get_name.output}")
        assert_skill_refusal_handoff(
            path_bad_get_name.metadata,
            "get_skill path-shaped name",
            source="get_skill",
            mutation="skill_read",
            reason="invalid_name",
            read_only=True,
        )
        for value in ["/var/folders/zc/jarvis-skill-name", "/tmp/jarvis-skill-name"]:
            path_bad_temp_get_name = runtime.registry.get("get_skill").handler({"name": value})
            if path_bad_temp_get_name.ok or path_bad_temp_get_name.metadata.get("reason") != "invalid_name":
                raise SystemExit(f"get_skill temp-root name missed safe refusal metadata: {path_bad_temp_get_name.metadata}")
            if path_bad_temp_get_name.metadata.get("raw_name") != "<local-path>":
                raise SystemExit(f"get_skill leaked temp-root path in raw name metadata: {path_bad_temp_get_name.metadata}")
            assert_no_local_path(path_bad_temp_get_name.output, "get_skill temp-root refusal output")

        missing_delete = runtime.registry.get("delete_skill").handler({"name": ""})
        if missing_delete.ok or missing_delete.metadata.get("raw_name") != "" or missing_delete.metadata.get("requires_approval") is not True:
            raise SystemExit(f"delete_skill missing name missed bounded raw name/approval metadata: {missing_delete.metadata}")
        assert_skill_refusal_handoff(
            missing_delete.metadata,
            "delete_skill missing name",
            source="delete_skill",
            mutation="skill_delete",
            reason="missing_name",
            read_only=False,
            requires_approval=True,
        )
        path_bad_delete_name = runtime.registry.get("delete_skill").handler({"name": "/\x55sers/example/private/delete-skill"})
        if path_bad_delete_name.ok or path_bad_delete_name.metadata.get("reason") != "invalid_name":
            raise SystemExit("delete_skill path-shaped name missed safe refusal metadata.")
        if path_bad_delete_name.metadata.get("raw_name") != "<local-path>" or path_bad_delete_name.metadata.get("requires_approval") is not True:
            raise SystemExit(f"delete_skill leaked local path or missed approval metadata: {path_bad_delete_name.metadata}")
        if path_bad_delete_name.metadata.get("writes_files") or path_bad_delete_name.metadata.get("writes_skills"):
            raise SystemExit(f"delete_skill path-shaped name should not write: {path_bad_delete_name.metadata}")
        if "/\x55sers/" in path_bad_delete_name.output or "/private/" in path_bad_delete_name.output:
            raise SystemExit(f"delete_skill leaked local path in refusal output: {path_bad_delete_name.output}")
        assert_skill_refusal_handoff(
            path_bad_delete_name.metadata,
            "delete_skill path-shaped name",
            source="delete_skill",
            mutation="skill_delete",
            reason="invalid_name",
            read_only=False,
            requires_approval=True,
        )
        for value in ["/var/folders/zc/delete-skill", "/tmp/delete-skill"]:
            path_bad_temp_delete_name = runtime.registry.get("delete_skill").handler({"name": value})
            if path_bad_temp_delete_name.ok or path_bad_temp_delete_name.metadata.get("reason") != "invalid_name":
                raise SystemExit(f"delete_skill temp-root name missed safe refusal metadata: {path_bad_temp_delete_name.metadata}")
            if path_bad_temp_delete_name.metadata.get("raw_name") != "<local-path>" or path_bad_temp_delete_name.metadata.get("requires_approval") is not True:
                raise SystemExit(f"delete_skill leaked temp-root path or missed approval metadata: {path_bad_temp_delete_name.metadata}")
            assert_no_local_path(path_bad_temp_delete_name.output, "delete_skill temp-root refusal output")

        delete_resolver = runtime.registry.get("delete_skill").approval_argument_resolver
        if delete_resolver is None:
            raise SystemExit("delete_skill is missing its pre-approval target resolver")
        missing_delete_target = delete_resolver({"name": "Missing Skill"})
        if missing_delete_target.ok or missing_delete_target.metadata.get("reason") != "not_found":
            raise SystemExit(f"delete_skill missing target missed refusal metadata: {missing_delete_target.metadata}")
        if missing_delete_target.metadata.get("writes_files") or missing_delete_target.metadata.get("writes_skills"):
            raise SystemExit(f"delete_skill missing target should not write: {missing_delete_target.metadata}")
        assert_skill_refusal_handoff(
            missing_delete_target.metadata,
            "delete_skill missing target",
            source="delete_skill",
            mutation="skill_delete",
            reason="not_found",
            read_only=False,
            requires_approval=True,
        )

        missing_request = runtime.registry.get("skill_match_preview").handler({"request": ""})
        if missing_request.ok or missing_request.metadata.get("reason") != "missing_request":
            raise SystemExit("skill_match_preview missing request missed safe refusal metadata.")
        path_bad_preview = runtime.registry.get("skill_match_preview").handler({"request": "/private/tmp/jarvis-skill-preview"})
        if path_bad_preview.ok or path_bad_preview.metadata.get("reason") != "invalid_request":
            raise SystemExit("skill_match_preview path-shaped request missed safe refusal metadata.")
        if path_bad_preview.metadata.get("raw_request") != "<local-path>":
            raise SystemExit(f"skill_match_preview leaked local path in raw request metadata: {path_bad_preview.metadata}")
        if path_bad_preview.metadata.get("writes_files") or path_bad_preview.metadata.get("writes_skills") or path_bad_preview.metadata.get("executes_tools"):
            raise SystemExit(f"skill_match_preview path-shaped request should stay read-only: {path_bad_preview.metadata}")
        if "/\x55sers/" in path_bad_preview.output or "/private/" in path_bad_preview.output:
            raise SystemExit(f"skill_match_preview leaked local path in refusal output: {path_bad_preview.output}")
        for value in ["/var/folders/zc/jarvis-skill-preview", "/tmp/jarvis-skill-preview"]:
            path_bad_temp_preview = runtime.registry.get("skill_match_preview").handler({"request": value})
            if path_bad_temp_preview.ok or path_bad_temp_preview.metadata.get("reason") != "invalid_request":
                raise SystemExit(f"skill_match_preview temp-root request missed safe refusal metadata: {path_bad_temp_preview.metadata}")
            if path_bad_temp_preview.metadata.get("raw_request") != "<local-path>":
                raise SystemExit(f"skill_match_preview leaked temp-root path in raw request metadata: {path_bad_temp_preview.metadata}")
            assert_no_local_path(path_bad_temp_preview.output, "skill_match_preview temp-root refusal output")

        bounded = runtime.registry.get("save_skill").handler({"name": "n" * 500, "trigger": "t" * 800, "body": "body", "tags": "x" * 800})
        if not bounded.ok:
            raise SystemExit("save_skill should accept bounded long fields.")
        if bounded.metadata.get("name_chars") != 160 or bounded.metadata.get("body_chars") != 4:
            raise SystemExit("save_skill did not bound long name field.")
        assert_skill_write_receipt(bounded, root=root, label="bounded save_skill")


if __name__ == "__main__":
    main()
