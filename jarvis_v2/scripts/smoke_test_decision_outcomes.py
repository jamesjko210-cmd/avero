from __future__ import annotations

import sqlite3
from pathlib import Path
from tempfile import TemporaryDirectory

import jarvis_v2.tools.decisions as decisions_module
from jarvis_v2.agent.planner import RuleBasedPlanner
from jarvis_v2.agent.types import RiskLevel
from jarvis_v2.memory.decision_projection import (
    reconcile_decision_projection,
    reconcile_pending_decision_projections,
)
from jarvis_v2.memory.store import MemoryStore, decision_outcome_summary_safety_reason
from jarvis_v2.scripts.test_runtime import make_temp_runtime


def _outcome_count(runtime, decision_id: int = 1) -> int:
    return len(runtime.store.list_decision_outcomes(decision_id, limit=100))


def _decision_note(runtime, decision_id: int = 1) -> Path:
    matches = list(
        (runtime.vault.root_path / "Decisions").glob(f"{decision_id:04d} *.md")
    )
    if len(matches) != 1:
        raise SystemExit(f"expected one decision note, found {matches}")
    return matches[0]


def _create_decision(runtime) -> None:
    result = runtime.handle(
        "record decision choose the safer design",
        request_token="decision-outcome-create",
    )
    if not result.tool_results or not result.tool_results[0].ok:
        raise SystemExit(f"decision setup failed: {result.response}")


def test_route_and_end_to_end() -> None:
    planner = RuleBasedPlanner()
    plan = planner.plan("decision outcome #42: the rollout reduced failures")
    if [(action.tool_name, action.args) for action in plan.actions] != [
        (
            "record_decision_outcome",
            {"decision_id": 42, "summary": "the rollout reduced failures"},
        )
    ]:
        raise SystemExit(f"decision outcome route drifted: {plan.actions}")

    compound = planner.plan(
        "decision outcome #42: the rollout worked and then show decisions"
    )
    if [(action.tool_name, action.args) for action in compound.actions] != [
        (
            "record_decision_outcome",
            {"decision_id": 42, "summary": "the rollout worked"},
        )
    ]:
        raise SystemExit(f"decision outcome bypassed the compound-command guard: {compound.actions}")

    for invalid in (
        "decision outcome #0: no",
        "decision outcome #1:",
        "decision outcome #1: " + ("x" * 4001),
    ):
        rejected = planner.plan(invalid)
        if rejected.actions or rejected.needs_model:
            raise SystemExit(f"invalid decision outcome did not fail closed: {invalid!r}")

    with TemporaryDirectory(prefix="jarvis-decision-outcome-e2e-") as temp:
        runtime = make_temp_runtime(Path(temp))
        _create_decision(runtime)
        tool = runtime.registry.get("record_decision_outcome")
        if tool.risk is not RiskLevel.LOCAL_SAFE or tool.argument_contract is None:
            raise SystemExit("decision outcome tool lost its local-safe typed contract")

        result = runtime.handle(
            "decision outcome #1: the rollout reduced failures",
            request_token="decision-outcome-first",
        )
        if len(result.tool_results) != 1 or not result.tool_results[0].ok:
            raise SystemExit(f"decision outcome did not complete: {result.response}")
        metadata = result.tool_results[0].metadata
        if (
            metadata.get("provenance") != "user_reported"
            or metadata.get("verified") is not False
            or metadata.get("verification_status") != "not_verified"
            or metadata.get("projection_pending") is not False
            or metadata.get("requires_approval") is not False
            or metadata.get("queues_approval") is not False
            or metadata.get("writes_database") is not True
            or metadata.get("writes_notes") is not True
            or metadata.get("writes_memory") is not False
        ):
            raise SystemExit(f"decision outcome receipt was not truthful: {metadata}")
        if runtime.store.list_pending_approvals(status=""):
            raise SystemExit("local decision outcome unexpectedly created an approval")

        rows = runtime.store.list_decision_outcomes(1, limit=100)
        decision = runtime.store.get_decision(1)
        if (
            len(rows) != 1
            or rows[0]["summary"] != "the rollout reduced failures"
            or rows[0]["provenance"] != "user_reported"
            or decision is None
            or decision["revision"] != 2
            or rows[0]["decision_revision"] != 2
        ):
            raise SystemExit("decision outcome source graph diverged")

        note = _decision_note(runtime).read_text(encoding="utf-8")
        for expected in (
            "## Outcomes",
            "Provenance: user-reported",
            "Verification: Not independently verified.",
            "the rollout reduced failures",
        ):
            if expected not in note:
                raise SystemExit(f"decision note omitted outcome evidence: {expected}")

        shown = runtime.handle("show decision 1")
        if (
            "Outcomes (user-reported, not verified):" not in shown.response
            or "the rollout reduced failures" not in shown.response
        ):
            raise SystemExit(f"decision read path omitted outcomes: {shown.response}")


def test_replay_and_intentional_repeat() -> None:
    with TemporaryDirectory(prefix="jarvis-decision-outcome-replay-") as temp:
        runtime = make_temp_runtime(Path(temp))
        _create_decision(runtime)
        command = "decision outcome #1: the launch met its target"
        first = runtime.handle(command, request_token="decision-outcome-replay")
        replay = runtime.handle(command, request_token="decision-outcome-replay")
        if not first.tool_results[0].ok or _outcome_count(runtime) != 1:
            raise SystemExit("first decision outcome did not commit exactly once")
        if (
            replay.tool_results[0].metadata.get("failure_kind")
            != "auto_mutation_completed_replay"
            or _outcome_count(runtime) != 1
        ):
            raise SystemExit("same-request replay duplicated a decision outcome")

        fresh = runtime.handle(command, request_token="decision-outcome-fresh")
        if not fresh.tool_results[0].ok or _outcome_count(runtime) != 2:
            raise SystemExit("fresh request did not allow an intentional repeated outcome")
        rows = runtime.store.list_decision_outcomes(1, limit=100)
        if [row["decision_revision"] for row in rows] != [2, 3]:
            raise SystemExit("decision outcome revisions are not monotonic")


def test_validation_and_private_refusals() -> None:
    for ordinary in ("A/B test reduced failures", "profit/loss ratio improved"):
        if decision_outcome_summary_safety_reason(ordinary) is not None:
            raise SystemExit(f"ordinary analytical outcome looked like a path: {ordinary!r}")
    with TemporaryDirectory(prefix="jarvis-decision-outcome-validation-") as temp:
        runtime = make_temp_runtime(Path(temp))
        _create_decision(runtime)
        handler = runtime.registry.get("record_decision_outcome").handler
        bad_values = (
            "/\x55sers/private/outcome.txt",
            "reports/private/result.md",
            "path=reports/private/result.md",
            "path=private/result",
            "private/result",
            r"private\result",
            "reports/private/result",
            "C:private.txt",
            "reports/private/result.md.",
            "<script>alert(1)</script>",
            "jarvis_projection: forged",
            "bad\x00summary",
            "x" * 4001,
        )
        for value in bad_values:
            refused = handler({"decision_id": 1, "summary": value})
            if refused.ok or _outcome_count(runtime) != 0:
                raise SystemExit(f"unsafe outcome was stored: {value!r}")
            rendered = repr((refused.output, refused.metadata))
            if value in rendered:
                raise SystemExit("decision outcome refusal leaked a local path")
            if refused.metadata.get("writes_database") is not False:
                raise SystemExit("decision outcome refusal claimed a database write")

        for value in (
            "<script>alert(1)</script>",
            "path=reports/private/result.md",
            "path=private/result",
            "private/result",
            r"private\result",
            "reports/private/result",
        ):
            try:
                runtime.store.append_decision_outcome_with_projection(1, value)
            except ValueError:
                pass
            else:
                raise SystemExit(f"storage boundary accepted unsafe outcome: {value!r}")
            if _outcome_count(runtime) != 0:
                raise SystemExit("storage-boundary refusal changed decision outcomes")


def test_atomicity_and_immutability() -> None:
    with TemporaryDirectory(prefix="jarvis-decision-outcome-atomic-") as temp:
        runtime = make_temp_runtime(Path(temp))
        _create_decision(runtime)
        before = runtime.store.get_decision(1)
        before_job = runtime.store.get_decision_projection_job(1)
        with runtime.store.connect() as conn:
            conn.execute(
                "CREATE TRIGGER fail_outcome_insert BEFORE INSERT ON decision_outcomes "
                "BEGIN SELECT RAISE(ABORT, 'forced outcome failure'); END"
            )
        try:
            runtime.store.append_decision_outcome_with_projection(1, "must roll back")
        except sqlite3.IntegrityError:
            pass
        else:
            raise SystemExit("forced outcome insert failure did not raise")
        after = runtime.store.get_decision(1)
        after_job = runtime.store.get_decision_projection_job(1)
        if (
            before is None
            or after is None
            or before_job is None
            or after_job is None
            or after["revision"] != before["revision"]
            or after_job["decision_revision"] != before_job["decision_revision"]
            or _outcome_count(runtime) != 0
        ):
            raise SystemExit("failed outcome insert left a partial revision or projection")
        with runtime.store.connect() as conn:
            conn.execute("DROP TRIGGER fail_outcome_insert")

        try:
            with runtime.store.connect() as conn:
                conn.execute(
                    "INSERT INTO decision_outcomes("
                    "decision_id, decision_revision, summary, provenance, created_at"
                    ") VALUES (1, 2, 'revision drift', 'user_reported', "
                    "'2026-07-13T00:00:00Z')"
                )
        except sqlite3.IntegrityError:
            pass
        else:
            raise SystemExit("decision outcome accepted a revision ahead of its parent")

        target = runtime.store.append_decision_outcome_with_projection(1, "immutable result")
        if target is None:
            raise SystemExit("decision outcome source append unexpectedly missed its decision")
        for statement in (
            "UPDATE decision_outcomes SET summary = 'rewritten' WHERE id = 1",
            "DELETE FROM decision_outcomes WHERE id = 1",
        ):
            try:
                with runtime.store.connect() as conn:
                    conn.execute(statement)
            except sqlite3.IntegrityError:
                pass
            else:
                raise SystemExit(f"append-only decision outcome accepted: {statement}")

        with runtime.store.connect() as conn:
            conn.executescript(
                """
                DROP TRIGGER decision_outcomes_current_revision_insert;
                DROP TRIGGER decision_outcomes_immutable_update;
                DROP TRIGGER decision_outcomes_immutable_delete;
                CREATE TRIGGER decision_outcomes_current_revision_insert
                BEFORE INSERT ON decision_outcomes
                WHEN NEW.decision_revision = 2 AND NOT EXISTS (
                    SELECT 1 FROM decisions
                    WHERE id = NEW.decision_id AND revision = NEW.decision_revision
                )
                BEGIN SELECT RAISE(ABORT, 'decision outcome revision is not current'); END;
                CREATE TRIGGER decision_outcomes_immutable_update
                BEFORE UPDATE OF summary ON decision_outcomes
                BEGIN SELECT RAISE(ABORT, 'decision outcomes are immutable'); END;
                CREATE TRIGGER decision_outcomes_immutable_delete
                BEFORE DELETE ON decision_outcomes
                WHEN OLD.summary = 'guard probe'
                BEGIN SELECT RAISE(ABORT, 'decision outcomes are immutable'); END;
                """
            )
        try:
            runtime.store.init()
        except RuntimeError:
            pass
        else:
            raise SystemExit("startup accepted no-op append-only outcome triggers")

    with TemporaryDirectory(prefix="jarvis-decision-outcome-invalid-startup-") as temp:
        db_path = Path(temp) / "jarvis.db"
        with sqlite3.connect(db_path) as conn:
            conn.executescript(
                """
                PRAGMA foreign_keys = ON;
                CREATE TABLE decisions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT NOT NULL,
                    rationale TEXT NOT NULL DEFAULT '',
                    impact TEXT NOT NULL DEFAULT '',
                    status TEXT NOT NULL DEFAULT 'active',
                    revision INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE decision_outcomes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    decision_id INTEGER NOT NULL REFERENCES decisions(id) ON DELETE RESTRICT,
                    decision_revision INTEGER NOT NULL,
                    summary TEXT NOT NULL,
                    provenance TEXT NOT NULL DEFAULT 'user_reported',
                    created_at TEXT NOT NULL,
                    UNIQUE(decision_id, decision_revision)
                );
                INSERT INTO decisions(
                    title, rationale, impact, status, revision, created_at, updated_at
                ) VALUES (
                    'legacy drift', '', '', 'active', 1,
                    '2026-07-13T00:00:00Z', '2026-07-13T00:00:00Z'
                );
                INSERT INTO decision_outcomes(
                    decision_id, decision_revision, summary, provenance, created_at
                ) VALUES (
                    1, 2, 'ahead of parent', 'user_reported', '2026-07-13T00:00:00Z'
                );
                """
            )
        try:
            MemoryStore(db_path).init()
        except RuntimeError:
            pass
        else:
            raise SystemExit("startup accepted revision-inconsistent immutable outcomes")

    with TemporaryDirectory(prefix="jarvis-decision-outcome-invalid-parent-") as temp:
        db_path = Path(temp) / "jarvis.db"
        with sqlite3.connect(db_path) as conn:
            conn.executescript(
                """
                CREATE TABLE decisions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT NOT NULL,
                    rationale TEXT NOT NULL DEFAULT '',
                    impact TEXT NOT NULL DEFAULT '',
                    status TEXT NOT NULL DEFAULT 'active',
                    revision INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                INSERT INTO decisions(
                    title, rationale, impact, status, revision, created_at, updated_at
                ) VALUES (
                    'bad parent revision', '', '', 'active', 2.5,
                    '2026-07-13T00:00:00Z', '2026-07-13T00:00:00Z'
                );
                """
            )
        try:
            MemoryStore(db_path).init()
        except RuntimeError:
            pass
        else:
            raise SystemExit("startup accepted a non-integer decision revision")

    with TemporaryDirectory(prefix="jarvis-decision-outcome-legacy-parent-") as temp:
        db_path = Path(temp) / "jarvis.db"
        with sqlite3.connect(db_path) as conn:
            conn.executescript(
                """
                CREATE TABLE decisions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT NOT NULL,
                    rationale TEXT NOT NULL DEFAULT '',
                    impact TEXT NOT NULL DEFAULT '',
                    status TEXT NOT NULL DEFAULT 'active',
                    revision INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                INSERT INTO decisions(
                    title, rationale, impact, status, revision, created_at, updated_at
                ) VALUES (
                    'legacy parent revision', '', '', 'active', 1,
                    '2026-07-13T00:00:00Z', '2026-07-13T00:00:00Z'
                );
                """
            )
        store = MemoryStore(db_path)
        store.init()
        try:
            with store.connect() as conn:
                conn.execute("UPDATE decisions SET revision = 2.5 WHERE id = 1")
        except sqlite3.IntegrityError:
            pass
        else:
            raise SystemExit("legacy decision schema accepted a non-integer future revision")


def test_projection_failure_repairs_without_replay() -> None:
    with TemporaryDirectory(prefix="jarvis-decision-outcome-repair-") as temp:
        runtime = make_temp_runtime(Path(temp))
        _create_decision(runtime)
        original = runtime.vault.write_decision_with_evidence

        def fail_write(*_args, **_kwargs):
            raise OSError("private projection failure")

        runtime.vault.write_decision_with_evidence = fail_write  # type: ignore[method-assign]
        try:
            result = runtime.handle(
                "decision outcome #1: committed before mirror",
                request_token="decision-outcome-projection-failure",
            )
        finally:
            runtime.vault.write_decision_with_evidence = original  # type: ignore[method-assign]
        pending = result.tool_results[0]
        if (
            not pending.ok
            or pending.metadata.get("projection_pending") is not True
            or pending.metadata.get("do_not_record_again") is not True
            or pending.metadata.get("auto_mutation_outcome_uncertain") is True
            or pending.metadata.get("writes_files") is not False
            or pending.metadata.get("writes_notes") is not False
            or pending.metadata.get("path_display") != ""
            or pending.metadata["decision_mutation_handoff"]["boundaries"].get(
                "writes_files"
            )
            is not False
            or pending.metadata["decision_mutation_handoff"]["boundaries"].get(
                "writes_notes"
            )
            is not False
            or _outcome_count(runtime) != 1
            or "do not record again" not in pending.output.lower()
        ):
            raise SystemExit(f"projection failure receipt was unsafe: {pending}")

        replay = runtime.handle(
            "decision outcome #1: committed before mirror",
            request_token="decision-outcome-projection-failure",
        )
        if (
            replay.tool_results[0].metadata.get("failure_kind")
            != "auto_mutation_completed_replay"
            or _outcome_count(runtime) != 1
        ):
            raise SystemExit("known committed outcome was not replay-safe while projection was pending")

        repaired = reconcile_decision_projection(runtime.store, runtime.vault, 1)
        if repaired.status != "completed" or _outcome_count(runtime) != 1:
            raise SystemExit(f"decision projection repair did not converge: {repaired}")
        note = _decision_note(runtime).read_text(encoding="utf-8")
        if "committed before mirror" not in note:
            raise SystemExit("decision projection repair omitted the committed outcome")


def test_legacy_projection_and_supersession_recovery() -> None:
    with TemporaryDirectory(prefix="jarvis-decision-outcome-legacy-") as temp:
        runtime = make_temp_runtime(Path(temp))
        _create_decision(runtime)
        decision = runtime.store.get_decision(1)
        if decision is None:
            raise SystemExit("legacy decision fixture disappeared")
        runtime.vault.write_decision(decision)
        result = runtime.handle(
            "decision outcome #1: legacy note upgraded",
            request_token="decision-outcome-legacy-upgrade",
        )
        if not result.tool_results[0].ok or result.tool_results[0].metadata.get(
            "projection_pending"
        ):
            raise SystemExit(f"legacy decision note could not be upgraded: {result.response}")
        note = _decision_note(runtime).read_text(encoding="utf-8")
        if "jarvis_projection: decision" not in note or "legacy note upgraded" not in note:
            raise SystemExit("legacy decision projection did not gain evidence and outcome")

    with TemporaryDirectory(prefix="jarvis-decision-outcome-legacy-drift-") as temp:
        runtime = make_temp_runtime(Path(temp))
        _create_decision(runtime)
        decision = runtime.store.get_decision(1)
        if decision is None:
            raise SystemExit("legacy drift fixture disappeared")
        path = runtime.vault.write_decision(decision)
        legacy = path.read_text(encoding="utf-8") + "\nManual content that must survive.\n"
        path.write_text(legacy, encoding="utf-8")
        result = runtime.handle(
            "decision outcome #1: source committed without clobbering",
            request_token="decision-outcome-legacy-drift",
        )
        if (
            not result.tool_results[0].ok
            or result.tool_results[0].metadata.get("projection_pending") is not True
            or _outcome_count(runtime) != 1
            or path.read_text(encoding="utf-8") != legacy
        ):
            raise SystemExit("legacy content drift was overwritten or outcome custody was lost")

    with TemporaryDirectory(prefix="jarvis-decision-outcome-superseded-") as temp:
        runtime = make_temp_runtime(Path(temp))
        _create_decision(runtime)
        original_reconcile = decisions_module.reconcile_decision_projection
        injected = False

        def append_newer_before_reconcile(store, vault, target):
            nonlocal injected
            if not injected:
                injected = True
                store.append_decision_outcome_with_projection(1, "newer concurrent outcome")
            return original_reconcile(store, vault, target)

        decisions_module.reconcile_decision_projection = append_newer_before_reconcile
        try:
            result = runtime.handle(
                "decision outcome #1: first concurrent outcome",
                request_token="decision-outcome-superseded",
            )
        finally:
            decisions_module.reconcile_decision_projection = original_reconcile
        metadata = result.tool_results[0].metadata
        if (
            not result.tool_results[0].ok
            or metadata.get("decision_projection_status") != "superseded"
            or metadata.get("projection_pending") is not False
            or _outcome_count(runtime) != 2
            or not runtime.store.current_decision_projection_completion(1)
        ):
            raise SystemExit(f"superseded completed projection was reported as pending: {result}")

    with TemporaryDirectory(prefix="jarvis-decision-outcome-stale-identity-") as temp:
        runtime = make_temp_runtime(Path(temp))
        _create_decision(runtime)
        with runtime.store.connect() as conn:
            conn.execute(
                "UPDATE decision_projection_jobs SET store_identity = ? WHERE decision_id = 1",
                ("f" * 32,),
            )
        if runtime.store.count_pending_decision_projection_jobs() != 1:
            raise SystemExit("stale-identity decision job was hidden from pending health")
        repaired = reconcile_pending_decision_projections(runtime.store, runtime.vault, limit=10)
        if (
            repaired.completed != 1
            or repaired.pending != 0
            or not runtime.store.current_decision_projection_completion(1)
        ):
            raise SystemExit(f"batch repair ignored a stale-identity decision job: {repaired}")


def test_status_projection_pending_is_known_and_replay_safe() -> None:
    with TemporaryDirectory(prefix="jarvis-decision-status-pending-") as temp:
        runtime = make_temp_runtime(Path(temp))
        _create_decision(runtime)
        original = runtime.vault.write_decision_with_evidence

        def fail_write(*_args, **_kwargs):
            raise OSError("private status projection failure")

        runtime.vault.write_decision_with_evidence = fail_write  # type: ignore[method-assign]
        try:
            result = runtime.handle(
                "decision 1 superseded",
                request_token="decision-status-projection-failure",
            )
        finally:
            runtime.vault.write_decision_with_evidence = original  # type: ignore[method-assign]
        item = result.tool_results[0]
        row = runtime.store.get_decision(1)
        if (
            not item.ok
            or row is None
            or row["status"] != "superseded"
            or row["revision"] != 2
            or item.metadata.get("projection_pending") is not True
            or item.metadata.get("auto_mutation_outcome_uncertain") is True
            or item.metadata.get("writes_files") is not False
            or item.metadata.get("writes_notes") is not False
            or item.metadata.get("path_display") != ""
            or "do not repeat" not in item.output.lower()
        ):
            raise SystemExit(f"known pending decision status receipt was unsafe: {item}")
        replay = runtime.handle(
            "decision 1 superseded",
            request_token="decision-status-projection-failure",
        )
        if (
            replay.tool_results[0].metadata.get("failure_kind")
            != "auto_mutation_completed_replay"
            or runtime.store.get_decision(1)["revision"] != 2
        ):
            raise SystemExit("known committed decision status was not replay-safe")
        repaired = reconcile_decision_projection(runtime.store, runtime.vault, 1)
        if repaired.status != "completed":
            raise SystemExit(f"pending decision status did not repair: {repaired}")


def main() -> None:
    test_route_and_end_to_end()
    test_replay_and_intentional_repeat()
    test_validation_and_private_refusals()
    test_atomicity_and_immutability()
    test_projection_failure_repairs_without_replay()
    test_legacy_projection_and_supersession_recovery()
    test_status_projection_pending_is_known_and_replay_safe()
    print("Decision outcome smoke test passed.")


if __name__ == "__main__":
    main()
