from __future__ import annotations

import sqlite3
from pathlib import Path
from tempfile import TemporaryDirectory

from jarvis_v2.agent.runtime import JarvisRuntime
from jarvis_v2.agent.types import RiskLevel
from jarvis_v2.config import JarvisConfig
from jarvis_v2.memory.decision_projection import reconcile_decision_projection
from jarvis_v2.memory.memory_projection import reconcile_memory_projection
from jarvis_v2.memory.obsidian import ObsidianVault
from jarvis_v2.memory.preference_projection import reconcile_preference_projection
from jarvis_v2.memory.store import (
    DecisionMutationTarget,
    DecisionRecord,
    MemoryRecord,
    MemoryStore,
    PreferenceMutationTarget,
    PreferenceRecord,
)
from jarvis_v2.tools.decisions import make_decision_tools
from jarvis_v2.tools.preferences import make_preference_tools


DECISION_GRAPH_TABLES = (
    "decisions",
    "memories",
    "decision_memory_links",
    "decision_projection_jobs",
    "memory_projection_jobs",
)
PREFERENCE_GRAPH_TABLES = (
    "preferences",
    "preference_identity_owners",
    "memories",
    "preference_memory_links",
    "preference_projection_jobs",
    "memory_projection_jobs",
)


def _setup(root: Path) -> tuple[MemoryStore, ObsidianVault]:
    store = MemoryStore(root / "store.sqlite")
    store.init()
    vault = ObsidianVault(root / "vault")
    vault.init()
    return store, vault


def _runtime_config(root: Path) -> JarvisConfig:
    return JarvisConfig(
        data_dir=root,
        db_path=root / "jarvis.sqlite",
        obsidian_vault=root / "Vault",
        obsidian_root="Jarvis",
        use_model_planner=False,
        watched_dirs=(root / "Watched",),
    )


def _state_changed(result) -> bool:
    metadata = result.metadata or {}
    handoff = metadata.get("preference_mutation_handoff") or {}
    return metadata.get("state_changed") is True or handoff.get("state_changed") is True


def _counts(store: MemoryStore, tables: tuple[str, ...]) -> dict[str, int]:
    with store.connect() as conn:
        return {
            table: int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
            for table in tables
        }


def _assert_one_graph(store: MemoryStore, tables: tuple[str, ...], label: str) -> None:
    counts = _counts(store, tables)
    if counts != {table: 1 for table in tables}:
        raise SystemExit(f"{label} did not commit one exact graph: {counts}")


def _assert_no_graph(store: MemoryStore, tables: tuple[str, ...], label: str) -> None:
    counts = _counts(store, tables)
    if any(counts.values()):
        raise SystemExit(f"{label} left a partial graph: {counts}")


def _assert_private(
    values: tuple[object, ...], forbidden: tuple[str, ...], label: str
) -> None:
    rendered = "\n".join(repr(value) for value in values)
    leaked = [value for value in forbidden if value and value in rendered]
    if leaked:
        raise SystemExit(f"{label} leaked source content or a local path: {leaked}")


def _rows(store: MemoryStore, query: str, params: tuple[object, ...] = ()):
    with store.connect() as conn:
        return list(conn.execute(query, params))


def _decision_target(store: MemoryStore, record: DecisionRecord) -> DecisionMutationTarget:
    target = store.record_decision_with_projections(record)
    if not isinstance(target, DecisionMutationTarget):
        raise SystemExit(
            "record_decision_with_projections returned the wrong target type: "
            f"{type(target).__name__}"
        )
    return target


def _preference_target(
    store: MemoryStore, record: PreferenceRecord
) -> PreferenceMutationTarget:
    target = store.set_preference_with_projections(record)
    if not isinstance(target, PreferenceMutationTarget):
        raise SystemExit(
            "set_preference_with_projections returned the wrong target type: "
            f"{type(target).__name__}"
        )
    return target


def _assert_decision_graph(
    store: MemoryStore,
    target: DecisionMutationTarget,
    *,
    expected_title: str,
) -> None:
    with store.connect() as conn:
        decision = conn.execute(
            "SELECT * FROM decisions WHERE id = ?", (target.decision_id,)
        ).fetchone()
        memory = conn.execute(
            "SELECT * FROM memories WHERE id = ?", (target.memory_id,)
        ).fetchone()
        link = conn.execute(
            "SELECT * FROM decision_memory_links WHERE decision_id = ?",
            (target.decision_id,),
        ).fetchone()
        decision_job = conn.execute(
            "SELECT * FROM decision_projection_jobs WHERE decision_id = ?",
            (target.decision_id,),
        ).fetchone()
        memory_job = conn.execute(
            "SELECT * FROM memory_projection_jobs WHERE memory_id = ?",
            (target.memory_id,),
        ).fetchone()
    if any(row is None for row in (decision, memory, link, decision_job, memory_job)):
        raise SystemExit("decision mutation omitted a source, link, memory, or job")
    if (
        decision["title"] != expected_title
        or memory["category"] != "decisions"
        or memory["source"] != "decision-log"
        or link["memory_id"] != target.memory_id
        or decision_job["state"] != "pending"
        or decision_job["decision_revision"]
        != target.decision_projection_target.revision
        or decision_job["source_digest"]
        != target.decision_projection_target.source_digest
        or memory_job["state"] != "pending"
        or memory_job["operation"] != "publish"
        or memory_job["memory_revision"] != target.memory_projection_target.revision
        or memory_job["source_digest"] != target.memory_projection_target.source_digest
    ):
        raise SystemExit("decision mutation graph diverged from its returned targets")


def _assert_preference_graph(
    store: MemoryStore,
    target: PreferenceMutationTarget,
    *,
    expected_value: str,
) -> None:
    with store.connect() as conn:
        preference = conn.execute(
            "SELECT * FROM preferences WHERE id = ?", (target.preference_id,)
        ).fetchone()
        memory = conn.execute(
            "SELECT * FROM memories WHERE id = ?", (target.memory_id,)
        ).fetchone()
        owner = conn.execute(
            "SELECT * FROM preference_identity_owners WHERE preference_id = ?",
            (target.preference_id,),
        ).fetchone()
        link = conn.execute(
            "SELECT * FROM preference_memory_links WHERE preference_id = ?",
            (target.preference_id,),
        ).fetchone()
        preference_job = conn.execute(
            "SELECT * FROM preference_projection_jobs WHERE singleton_id = 1"
        ).fetchone()
        memory_job = conn.execute(
            "SELECT * FROM memory_projection_jobs WHERE memory_id = ?",
            (target.memory_id,),
        ).fetchone()
    if any(
        row is None
        for row in (preference, memory, owner, link, preference_job, memory_job)
    ):
        raise SystemExit("preference mutation omitted an owner, source, link, memory, or job")
    if (
        preference["value"] != expected_value
        or expected_value not in memory["body"]
        or link["memory_id"] != target.memory_id
        or preference_job["state"] != "pending"
        or preference_job["generation"]
        != target.preference_projection_target.generation
        or preference_job["source_digest"]
        != target.preference_projection_target.source_digest
        or memory_job["state"] != "pending"
        or memory_job["operation"] != "publish"
        or memory_job["memory_revision"] != target.memory_projection_target.revision
        or memory_job["source_digest"] != target.memory_projection_target.source_digest
    ):
        raise SystemExit("preference mutation graph diverged from its returned targets")


def test_atomic_graphs_and_late_sql_rollback() -> None:
    with TemporaryDirectory(prefix="jarvis-decision-preference-atomic-") as temp:
        root = Path(temp)

        decision_store, decision_vault = _setup(root / "decision-success")
        files_before = tuple(decision_vault.root_path.rglob("*.md"))
        decision = _decision_target(
            decision_store,
            DecisionRecord(
                "PRIVATE-DECISION-TITLE-ATOMIC",
                "PRIVATE-DECISION-RATIONALE-ATOMIC",
                "PRIVATE-DECISION-IMPACT-ATOMIC",
            ),
        )
        _assert_one_graph(decision_store, DECISION_GRAPH_TABLES, "decision mutation")
        _assert_decision_graph(
            decision_store, decision, expected_title="PRIVATE-DECISION-TITLE-ATOMIC"
        )
        if tuple(decision_vault.root_path.rglob("*.md")) != files_before:
            raise SystemExit("decision store primitive wrote projection files")

        decision_rollback, _ = _setup(root / "decision-rollback")
        with decision_rollback.connect() as conn:
            conn.execute(
                """
                CREATE TRIGGER reject_decision_projection_job
                BEFORE INSERT ON decision_projection_jobs
                BEGIN
                    SELECT RAISE(ABORT, 'injected decision custody failure');
                END
                """
            )
        try:
            _decision_target(
                decision_rollback,
                DecisionRecord("ROLLBACK-TITLE", "ROLLBACK-RATIONALE", "ROLLBACK-IMPACT"),
            )
        except sqlite3.DatabaseError:
            pass
        else:
            raise SystemExit("decision late SQL failure did not abort")
        _assert_no_graph(
            decision_rollback, DECISION_GRAPH_TABLES, "decision late SQL failure"
        )

        preference_store, preference_vault = _setup(root / "preference-success")
        files_before = tuple(preference_vault.root_path.rglob("*.md"))
        preference = _preference_target(
            preference_store,
            PreferenceRecord(
                "PRIVATE-PREFERENCE-KEY-ATOMIC",
                "PRIVATE-PREFERENCE-VALUE-ATOMIC",
                "PRIVATE-PREFERENCE-CATEGORY-ATOMIC",
            ),
        )
        _assert_one_graph(
            preference_store, PREFERENCE_GRAPH_TABLES, "preference mutation"
        )
        _assert_preference_graph(
            preference_store,
            preference,
            expected_value="PRIVATE-PREFERENCE-VALUE-ATOMIC",
        )
        if tuple(preference_vault.root_path.rglob("*.md")) != files_before:
            raise SystemExit("preference store primitive wrote projection files")

        preference_rollback, _ = _setup(root / "preference-rollback")
        with preference_rollback.connect() as conn:
            conn.execute(
                """
                CREATE TRIGGER reject_preference_projection_job
                BEFORE INSERT ON preference_projection_jobs
                BEGIN
                    SELECT RAISE(ABORT, 'injected preference custody failure');
                END
                """
            )
        try:
            _preference_target(
                preference_rollback,
                PreferenceRecord("ROLLBACK-KEY", "ROLLBACK-VALUE", "ROLLBACK-CATEGORY"),
            )
        except sqlite3.DatabaseError:
            pass
        else:
            raise SystemExit("preference late SQL failure did not abort")
        _assert_no_graph(
            preference_rollback,
            PREFERENCE_GRAPH_TABLES,
            "preference late SQL failure",
        )


def test_specialized_publish_failure_repairs_without_duplication() -> None:
    with TemporaryDirectory(prefix="jarvis-decision-preference-repair-") as temp:
        root = Path(temp)

        decision_store, decision_vault = _setup(root / "decision")
        decision = _decision_target(
            decision_store,
            DecisionRecord("Repair decision", "repair rationale", "repair impact"),
        )
        original_decision_write = decision_vault.write_decision_with_evidence

        def fail_decision_write(*_args, **_kwargs):
            raise OSError("injected decision note failure")

        decision_vault.write_decision_with_evidence = fail_decision_write
        failed = reconcile_decision_projection(
            decision_store, decision_vault, decision.decision_projection_target
        )
        decision_vault.write_decision_with_evidence = original_decision_write
        decision_job = decision_store.get_decision_projection_job(decision.decision_id)
        memory_job = decision_store.get_memory_projection_job(decision.memory_id)
        if (
            failed.status != "pending_error"
            or decision_job is None
            or decision_job["state"] != "pending"
            or decision_job["last_error_code"] != "vault_publish_failed"
            or memory_job is None
            or memory_job["state"] != "pending"
        ):
            raise SystemExit("decision note failure did not preserve a repairable full graph")
        _assert_one_graph(
            decision_store, DECISION_GRAPH_TABLES, "failed decision publication"
        )
        repaired = reconcile_decision_projection(
            decision_store, decision_vault, decision.decision_projection_target
        )
        repaired_memory = reconcile_memory_projection(
            decision_store,
            decision_vault,
            decision.memory_id,
            expected_operation=decision.memory_projection_target.operation,
            expected_revision=decision.memory_projection_target.revision,
            expected_source_digest=decision.memory_projection_target.source_digest,
        )
        if repaired.status != "completed" or repaired_memory.status != "completed":
            raise SystemExit(
                f"decision reconciliation did not repair both projections: "
                f"{repaired}, {repaired_memory}"
            )
        _assert_one_graph(
            decision_store, DECISION_GRAPH_TABLES, "repaired decision publication"
        )

        preference_store, preference_vault = _setup(root / "preference")
        preference = _preference_target(
            preference_store,
            PreferenceRecord("Repair key", "Repair value", "Repair category"),
        )
        original_preference_write = preference_vault.write_preferences_with_evidence

        def fail_preference_write(*_args, **_kwargs):
            raise OSError("injected preference note failure")

        preference_vault.write_preferences_with_evidence = fail_preference_write
        failed = reconcile_preference_projection(
            preference_store,
            preference_vault,
            preference.preference_projection_target,
        )
        preference_vault.write_preferences_with_evidence = original_preference_write
        preference_job = preference_store.get_preference_projection_job()
        memory_job = preference_store.get_memory_projection_job(preference.memory_id)
        if (
            failed.status != "pending_error"
            or preference_job is None
            or preference_job["state"] != "pending"
            or preference_job["last_error_code"] != "vault_publish_failed"
            or memory_job is None
            or memory_job["state"] != "pending"
        ):
            raise SystemExit("preference note failure did not preserve a repairable full graph")
        _assert_one_graph(
            preference_store,
            PREFERENCE_GRAPH_TABLES,
            "failed preference publication",
        )
        repaired = reconcile_preference_projection(
            preference_store,
            preference_vault,
            preference.preference_projection_target,
        )
        repaired_memory = reconcile_memory_projection(
            preference_store,
            preference_vault,
            preference.memory_id,
            expected_operation=preference.memory_projection_target.operation,
            expected_revision=preference.memory_projection_target.revision,
            expected_source_digest=preference.memory_projection_target.source_digest,
        )
        if repaired.status != "completed" or repaired_memory.status != "completed":
            raise SystemExit(
                f"preference reconciliation did not repair both projections: "
                f"{repaired}, {repaired_memory}"
            )
        _assert_one_graph(
            preference_store,
            PREFERENCE_GRAPH_TABLES,
            "repaired preference publication",
        )


def test_normalized_preference_reuses_one_canonical_memory() -> None:
    with TemporaryDirectory(prefix="jarvis-preference-normalized-") as temp:
        store, _ = _setup(Path(temp))
        first = _preference_target(
            store, PreferenceRecord("Theme", "light", "General")
        )
        second = _preference_target(
            store,
            PreferenceRecord("  ＴＨＥＭＥ  ", "dark", "  ＧＥＮＥＲＡＬ  "),
        )
        if (
            first.preference_id != second.preference_id
            or first.memory_id != second.memory_id
            or not second.changed
        ):
            raise SystemExit("normalized preference writes did not reuse canonical custody")
        _assert_one_graph(store, PREFERENCE_GRAPH_TABLES, "normalized preference writes")
        _assert_preference_graph(store, second, expected_value="dark")
        with store.connect() as conn:
            preference = conn.execute("SELECT * FROM preferences").fetchone()
            memory = conn.execute("SELECT * FROM memories").fetchone()
        if (
            preference is None
            or preference["revision"] != 2
            or preference["value"] != "dark"
            or memory is None
            or memory["revision"] != 2
            or memory["title"] != "Theme"
            or "Value: dark" not in memory["body"]
        ):
            raise SystemExit("normalized preference update lost latest canonical content")


def test_newer_status_projection_wins_for_decision_and_preference() -> None:
    with TemporaryDirectory(prefix="jarvis-decision-preference-status-race-") as temp:
        root = Path(temp)

        decision_store, decision_vault = _setup(root / "decision")
        decision = _decision_target(
            decision_store,
            DecisionRecord("Status race decision", "initial rationale", "initial impact"),
        )
        original_decision_write = decision_vault.write_decision_with_evidence
        decision_advanced = False
        latest_decision_target = None

        def advance_decision_status(snapshot, *, store_identity: str):
            nonlocal decision_advanced, latest_decision_target
            if not decision_advanced:
                decision_advanced = True
                latest_decision_target = decision_store.set_decision_status_with_projection(
                    decision.decision_id, "retired"
                )
            return original_decision_write(snapshot, store_identity=store_identity)

        decision_vault.write_decision_with_evidence = advance_decision_status
        decision_outcome = reconcile_decision_projection(
            decision_store, decision_vault, decision.decision_projection_target
        )
        decision_vault.write_decision_with_evidence = original_decision_write
        decision_job = decision_store.get_decision_projection_job(decision.decision_id)
        decision_row = decision_store.get_decision(decision.decision_id)
        decision_notes = list((decision_vault.root_path / "Decisions").glob("*.md"))
        if (
            decision_outcome.status != "superseded"
            or latest_decision_target is None
            or decision_job is None
            or decision_job["state"] != "completed"
            or decision_job["decision_revision"]
            != latest_decision_target.decision_projection_target.revision
            or decision_row is None
            or decision_row["status"] != "retired"
            or len(decision_notes) != 1
            or "status: retired" not in decision_notes[0].read_text(encoding="utf-8")
        ):
            raise SystemExit("newer decision status did not win the projection race")

        preference_store, preference_vault = _setup(root / "preference")
        preference = _preference_target(
            preference_store,
            PreferenceRecord("Status race key", "status race value", "General"),
        )
        original_preference_write = preference_vault.write_preferences_with_evidence
        preference_advanced = False
        latest_preference_target = None

        def advance_preference_status(
            rows, *, store_identity: str, generation: int
        ):
            nonlocal preference_advanced, latest_preference_target
            if not preference_advanced:
                preference_advanced = True
                latest_preference_target = (
                    preference_store.set_preference_status_with_projections(
                        preference.preference_id, "retired"
                    )
                )
            return original_preference_write(
                rows, store_identity=store_identity, generation=generation
            )

        preference_vault.write_preferences_with_evidence = advance_preference_status
        preference_outcome = reconcile_preference_projection(
            preference_store,
            preference_vault,
            preference.preference_projection_target,
        )
        preference_vault.write_preferences_with_evidence = original_preference_write
        preference_job = preference_store.get_preference_projection_job()
        preference_row = preference_store.get_preference(
            "Status race key", category="General"
        )
        preference_note = preference_vault.root_path / "Memory Tree" / "Preferences.md"
        if (
            preference_outcome.status != "superseded"
            or latest_preference_target is None
            or preference_job is None
            or preference_job["state"] != "completed"
            or preference_job["generation"]
            != latest_preference_target.preference_projection_target.generation
            or preference_row is None
            or preference_row["status"] != "retired"
            or not preference_note.exists()
            or "[retired]" not in preference_note.read_text(encoding="utf-8")
        ):
            raise SystemExit("newer preference status did not win the projection race")


def test_job_and_link_rows_are_content_free() -> None:
    with TemporaryDirectory(prefix="jarvis-decision-preference-privacy-") as temp:
        root = Path(temp)
        store, vault = _setup(root)
        decision_values = (
            "FORBIDDEN-DECISION-TITLE",
            "FORBIDDEN-DECISION-RATIONALE",
            "FORBIDDEN-DECISION-IMPACT",
        )
        preference_values = (
            "FORBIDDEN-PREFERENCE-KEY",
            "FORBIDDEN-PREFERENCE-VALUE",
        )
        decision = _decision_target(store, DecisionRecord(*decision_values))
        preference = _preference_target(
            store,
            PreferenceRecord(
                preference_values[0], preference_values[1], "privacy-category"
            ),
        )
        decision_outcome = reconcile_decision_projection(
            store, vault, decision.decision_projection_target
        )
        preference_outcome = reconcile_preference_projection(
            store, vault, preference.preference_projection_target
        )
        if decision_outcome.status != "completed" or preference_outcome.status != "completed":
            raise SystemExit("privacy fixture projections did not complete")

        tables = (
            "decision_projection_jobs",
            "decision_memory_links",
            "preference_projection_jobs",
            "preference_memory_links",
            "memory_projection_jobs",
        )
        forbidden_columns = {
            "decision_projection_jobs": {"title", "rationale", "impact", "path"},
            "decision_memory_links": {"title", "rationale", "impact", "path"},
            "preference_projection_jobs": {"key", "value", "path"},
            "preference_memory_links": {"key", "value", "path"},
            "memory_projection_jobs": {"body", "value", "rationale", "impact", "path"},
        }
        with store.connect() as conn:
            for table in tables:
                columns = {
                    str(row["name"]) for row in conn.execute(f"PRAGMA table_info({table})")
                }
                bad = columns & forbidden_columns[table]
                if bad:
                    raise SystemExit(f"{table} added raw source/path columns: {sorted(bad)}")
                rows = list(conn.execute(f"SELECT * FROM {table}"))
                _assert_private(
                    tuple(tuple(row) for row in rows),
                    decision_values + preference_values + (str(root),),
                    table,
                )


def test_runtime_startup_and_manual_repair_commands() -> None:
    with TemporaryDirectory(prefix="jarvis-decision-preference-startup-") as temp:
        root = Path(temp)
        store = MemoryStore(root / "jarvis.sqlite")
        store.init()
        decision = _decision_target(
            store,
            DecisionRecord("Startup decision", "startup rationale", "startup impact"),
        )
        preference = _preference_target(
            store,
            PreferenceRecord("Startup key", "startup value", "Startup category"),
        )
        before = _counts(
            store,
            (
                "decisions",
                "preferences",
                "memories",
                "decision_memory_links",
                "preference_memory_links",
            ),
        )
        runtime = JarvisRuntime(
            JarvisConfig(
                data_dir=root,
                db_path=root / "jarvis.sqlite",
                obsidian_vault=root / "Vault",
                obsidian_root="Jarvis",
                use_model_planner=False,
                watched_dirs=(root / "Watched",),
            )
        )
        after = _counts(runtime.store, tuple(before))
        if after != before:
            raise SystemExit(f"startup projection repair repeated source mutations: {after}")
        decision_job = runtime.store.get_decision_projection_job(decision.decision_id)
        preference_job = runtime.store.get_preference_projection_job()
        if (
            decision_job is None
            or decision_job["state"] != "completed"
            or preference_job is None
            or preference_job["state"] != "completed"
            or runtime.decision_projection_recovery_status.get("status") != "completed"
            or runtime.preference_projection_recovery_status.get("status") != "completed"
            or runtime.store.get_memory_projection_job(decision.memory_id)["state"]
            != "completed"
            or runtime.store.get_memory_projection_job(preference.memory_id)["state"]
            != "completed"
        ):
            raise SystemExit("runtime startup did not repair all reserved projections")

        for command, tool_name in (
            ("repair decision projections", "repair_decision_projections"),
            ("repair preference projections", "repair_preference_projections"),
        ):
            tool = runtime.registry.get(tool_name)
            if tool.risk is not RiskLevel.LOCAL_SAFE:
                raise SystemExit(f"{tool_name} is not local-safe")
            result = runtime.handle(command)
            if (
                not result.verified
                or len(result.tool_results) != 1
                or result.tool_results[0].tool_name != tool_name
                or not result.tool_results[0].ok
                or result.tool_results[0].metadata.get("writes_memory") is not False
                or result.tool_results[0].metadata.get("pending") != 0
                or result.metadata.get("approval_queue_delta") not in {None, 0}
            ):
                raise SystemExit(
                    f"manual repair alias did not run safely: {command}; "
                    f"verified={result.verified!r}; plan={result.plan!r}; "
                    f"tool_results={result.tool_results!r}; metadata={result.metadata!r}"
                )
        if _counts(runtime.store, tuple(before)) != before:
            raise SystemExit("manual projection repair repeated source mutations")


def test_runtime_startup_adopts_exact_legacy_graphs() -> None:
    with TemporaryDirectory(prefix="jarvis-legacy-decision-preference-startup-") as temp:
        root = Path(temp)
        config = _runtime_config(root)
        store = MemoryStore(config.db_path)
        store.init()
        vault = ObsidianVault(config.obsidian_vault, config.obsidian_root)
        vault.init()

        decision_id = store.add_decision(
            DecisionRecord("Legacy decision", "legacy rationale", "legacy impact")
        )
        decision_before = dict(store.get_decision(decision_id))
        decision_memory_id = store.add_memory(
            MemoryRecord(
                "decisions",
                "Legacy decision",
                "Legacy decision\n\nRationale: legacy rationale\n\nImpact: legacy impact",
                source="decision-log",
            )
        )
        vault.write_decision(store.get_decision(decision_id))

        preference_id = store.set_preference(
            PreferenceRecord("Theme", "legacy-light", "General")
        )
        preference_before = dict(store.get_preference_by_id(preference_id))
        preference_memory_id = store.add_memory(
            MemoryRecord(
                "preferences",
                "Theme",
                "Theme: legacy-light",
                source="preferences",
            )
        )
        vault.write_preferences(store.list_preferences(status=None))

        runtime = JarvisRuntime(config)
        if dict(runtime.store.get_decision(decision_id)) != decision_before:
            raise SystemExit("startup legacy decision custody mutated its source row")
        if dict(runtime.store.get_preference_by_id(preference_id)) != preference_before:
            raise SystemExit("startup legacy preference custody mutated its source row")
        with runtime.store.connect() as conn:
            decision_link = conn.execute(
                "SELECT memory_id FROM decision_memory_links WHERE decision_id = ?",
                (decision_id,),
            ).fetchone()
            preference_link = conn.execute(
                "SELECT memory_id FROM preference_memory_links WHERE preference_id = ?",
                (preference_id,),
            ).fetchone()
            memory_count = int(conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0])
        if (
            decision_link is None
            or int(decision_link["memory_id"]) != decision_memory_id
            or preference_link is None
            or int(preference_link["memory_id"]) != preference_memory_id
            or memory_count != 2
        ):
            raise SystemExit("startup did not adopt exact legacy memories without duplication")
        jobs = (
            runtime.store.get_decision_projection_job(decision_id),
            runtime.store.get_preference_projection_job(),
            runtime.store.get_memory_projection_job(decision_memory_id),
            runtime.store.get_memory_projection_job(preference_memory_id),
        )
        if any(job is None or job["state"] != "completed" for job in jobs):
            raise SystemExit("startup legacy custody did not complete all reserved jobs")


def test_legacy_preference_identity_collision_stays_visible() -> None:
    with TemporaryDirectory(prefix="jarvis-legacy-preference-collision-") as temp:
        root = Path(temp)
        config = _runtime_config(root)
        store = MemoryStore(config.db_path)
        store.init()
        now = "2026-01-01T00:00:00Z"
        with store.connect() as conn:
            conn.execute("DELETE FROM preference_identity_owners")
            conn.execute(
                "INSERT INTO preferences(category, key, value, status, revision, created_at, updated_at) "
                "VALUES (?, ?, ?, 'active', 1, ?, ?)",
                ("General", "Theme", "light", now, now),
            )
            conn.execute(
                "INSERT INTO preferences(category, key, value, status, revision, created_at, updated_at) "
                "VALUES (?, ?, ?, 'active', 1, ?, ?)",
                (" general ", "ＴＨＥＭＥ", "dark", now, now),
            )
        runtime = JarvisRuntime(config)
        with runtime.store.connect() as conn:
            source_count = int(conn.execute("SELECT COUNT(*) FROM preferences").fetchone()[0])
            owner_count = int(
                conn.execute("SELECT COUNT(*) FROM preference_identity_owners").fetchone()[0]
            )
            link_count = int(
                conn.execute("SELECT COUNT(*) FROM preference_memory_links").fetchone()[0]
            )
        status = runtime.preference_projection_recovery_status
        if (
            source_count != 2
            or owner_count != 1
            or link_count != 1
            or status.get("status") != "pending"
            or status.get("custody_conflicts") != 1
            or int(status.get("pending") or 0) < 1
        ):
            raise SystemExit(
                f"legacy normalized preference collision was hidden: {status}"
            )


def test_deleted_decision_link_blocks_success() -> None:
    with TemporaryDirectory(prefix="jarvis-decision-link-loss-") as temp:
        store, vault = _setup(Path(temp))
        record_decision, *_rest = make_decision_tools(store, vault)
        original_write = vault.write_decision_with_evidence

        def delete_link_after_publication(snapshot, *, store_identity: str):
            published = original_write(snapshot, store_identity=store_identity)
            with store.connect() as conn:
                conn.execute(
                    "DELETE FROM decision_memory_links WHERE decision_id = ?",
                    (int(snapshot["id"]),),
                )
            return published

        vault.write_decision_with_evidence = delete_link_after_publication
        result = record_decision(
            {"title": "Link loss", "rationale": "test", "impact": "test"}
        )
        vault.write_decision_with_evidence = original_write
        if result.ok or result.metadata.get("projection_pending") is not True:
            raise SystemExit("record_decision certified success after its custody link vanished")


def test_stale_handler_races_report_current_state() -> None:
    with TemporaryDirectory(prefix="jarvis-decision-preference-handler-races-") as temp:
        root = Path(temp)
        decision_store, decision_vault = _setup(root / "decision")
        decision = _decision_target(
            decision_store,
            DecisionRecord("Handler race", "initial", "initial"),
        )
        reconcile_decision_projection(
            decision_store, decision_vault, decision.decision_projection_target
        )
        _, _, _, _, set_decision_status = make_decision_tools(
            decision_store, decision_vault
        )
        original_decision_write = decision_vault.write_decision_with_evidence
        decision_raced = False

        def race_decision(snapshot, *, store_identity: str):
            nonlocal decision_raced
            if not decision_raced:
                decision_raced = True
                decision_store.set_decision_status_with_projection(
                    decision.decision_id, "retired"
                )
            return original_decision_write(snapshot, store_identity=store_identity)

        decision_vault.write_decision_with_evidence = race_decision
        decision_result = set_decision_status(
            {"decision_id": decision.decision_id, "status": "superseded"}
        )
        decision_vault.write_decision_with_evidence = original_decision_write
        if (
            decision_result.ok
            or decision_result.metadata.get("mutation_superseded") is not True
            or decision_result.metadata.get("status") != "retired"
            or "Current status: retired" not in decision_result.output
        ):
            raise SystemExit("stale decision handler race did not report current state")

        preference_store, preference_vault = _setup(root / "preference")
        preference = _preference_target(
            preference_store,
            PreferenceRecord("Race theme", "dark", "General"),
        )
        reconcile_preference_projection(
            preference_store,
            preference_vault,
            preference.preference_projection_target,
        )
        _, _, _, set_preference_status = make_preference_tools(
            preference_store, preference_vault
        )
        original_preference_write = preference_vault.write_preferences_with_evidence
        preference_raced = False

        def race_preference(rows, *, store_identity: str, generation: int):
            nonlocal preference_raced
            if not preference_raced:
                preference_raced = True
                preference_store.set_preference_status_with_projections(
                    preference.preference_id, "active"
                )
            return original_preference_write(
                rows, store_identity=store_identity, generation=generation
            )

        preference_vault.write_preferences_with_evidence = race_preference
        preference_result = set_preference_status(
            {"preference_id": preference.preference_id, "status": "retired"}
        )
        preference_vault.write_preferences_with_evidence = original_preference_write
        if (
            preference_result.ok
            or preference_result.metadata.get("mutation_superseded") is not True
            or preference_result.metadata.get("status") != "active"
            or "Current status: active" not in preference_result.output
        ):
            raise SystemExit("stale preference handler race did not report current state")


def test_same_key_categories_do_not_cross_adopt() -> None:
    with TemporaryDirectory(prefix="jarvis-preference-cross-category-") as temp:
        store, _vault = _setup(Path(temp))
        legacy_id = store.set_preference(PreferenceRecord("Theme", "blue", "UI"))
        legacy_memory_id = store.add_memory(
            MemoryRecord(
                "preferences", "Theme", "Theme: blue", source="preferences"
            )
        )
        current = _preference_target(
            store, PreferenceRecord("Theme", "quiet", "General")
        )
        with store.connect() as conn:
            legacy_memory = conn.execute(
                "SELECT * FROM memories WHERE id = ?", (legacy_memory_id,)
            ).fetchone()
            current_link = conn.execute(
                "SELECT memory_id FROM preference_memory_links WHERE preference_id = ?",
                (current.preference_id,),
            ).fetchone()
            legacy_link = conn.execute(
                "SELECT memory_id FROM preference_memory_links WHERE preference_id = ?",
                (legacy_id,),
            ).fetchone()
        if (
            legacy_memory is None
            or legacy_memory["body"] != "Theme: blue"
            or current_link is None
            or int(current_link["memory_id"]) == legacy_memory_id
            or legacy_link is not None
        ):
            raise SystemExit("same-key preference categories cross-adopted a legacy memory")


def test_manual_preference_section_is_preserved() -> None:
    with TemporaryDirectory(prefix="jarvis-preference-manual-section-") as temp:
        store, vault = _setup(Path(temp))
        store.set_preference(PreferenceRecord("Theme", "light", "General"))
        note = vault.write_preferences(store.list_preferences(status=None))
        exact_legacy = note.read_text(encoding="utf-8")
        with note.open("a", encoding="utf-8") as handle:
            handle.write("\n## Manual\n\nKEEP THIS\n")
        protected = note.read_text(encoding="utf-8")
        if not protected.startswith(exact_legacy):
            raise SystemExit("manual preference fixture is invalid")
        store.ensure_missing_preference_custody()
        target = store.ensure_current_preference_projection_job()
        outcome = reconcile_preference_projection(store, vault, target)
        job = store.get_preference_projection_job()
        if (
            outcome.status != "pending_error"
            or job is None
            or job["state"] != "pending"
            or note.read_text(encoding="utf-8") != protected
        ):
            raise SystemExit("preference reconciliation overwrote a manual legacy section")


def test_preference_noops_redaction_and_normalized_lookup() -> None:
    with TemporaryDirectory(prefix="jarvis-preference-handler-regressions-") as temp:
        store, vault = _setup(Path(temp))
        set_preference, _list_preferences, get_preference, set_status = (
            make_preference_tools(store, vault)
        )
        first = set_preference(
            {"key": "Theme", "value": "dark", "category": "General"}
        )
        if not first.ok:
            raise SystemExit(f"preference handler fixture failed: {first.output}")
        repeated = set_preference(
            {"key": "Theme", "value": "dark", "category": "General"}
        )
        preference_id = int(repeated.metadata["preference_id"])
        status_noop = set_status(
            {"preference_id": preference_id, "status": "active"}
        )
        if (
            not repeated.ok
            or _state_changed(repeated)
            or "already matches" not in repeated.output
            or not status_noop.ok
            or _state_changed(status_noop)
            or "already active" not in status_noop.output
        ):
            raise SystemExit("exact preference no-op reported a state change")

        local_path = "/\x55sers/example/private/project"
        redacted = set_preference(
            {"key": "Workspace", "value": local_path, "category": "General"}
        )
        if not redacted.ok or local_path in redacted.output or "<local-path>" not in redacted.output:
            raise SystemExit("path-shaped preference value leaked in handler output")

        normalized = get_preference({"key": "ＴＨＥＭＥ"})
        if not normalized.ok or "dark" not in normalized.output:
            raise SystemExit("uncategorized full-width preference lookup missed normalized identity")


def main() -> None:
    test_atomic_graphs_and_late_sql_rollback()
    test_specialized_publish_failure_repairs_without_duplication()
    test_normalized_preference_reuses_one_canonical_memory()
    test_newer_status_projection_wins_for_decision_and_preference()
    test_job_and_link_rows_are_content_free()
    test_runtime_startup_and_manual_repair_commands()
    test_runtime_startup_adopts_exact_legacy_graphs()
    test_legacy_preference_identity_collision_stays_visible()
    test_deleted_decision_link_blocks_success()
    test_stale_handler_races_report_current_state()
    test_same_key_categories_do_not_cross_adopt()
    test_manual_preference_section_is_preserved()
    test_preference_noops_redaction_and_normalized_lookup()
    print("decision/preference mutation atomicity smoke passed")


if __name__ == "__main__":
    main()
