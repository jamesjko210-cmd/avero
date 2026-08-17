from __future__ import annotations

import json
import sqlite3
import time
import traceback
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import Any, Callable
from unittest.mock import patch

from jarvis_v2.agent.planner import RuleBasedPlanner
from jarvis_v2.config import JarvisConfig
from jarvis_v2.memory.obsidian import ObsidianVault
from jarvis_v2.memory.store import (
    SQLITE_BUSY_TIMEOUT_SECONDS,
    STARTUP_RECOVERY_TABLE_SQL,
    STARTUP_RECOVERY_COMPONENTS,
    STARTUP_RECOVERY_STALE_AFTER_SECONDS,
    MemoryStore,
)
from jarvis_v2.scripts.startup import StartupRecoveryUnavailable
from jarvis_v2.tools.registry import ToolArgumentType, build_core_registry
from jarvis_v2.tools.startup_recovery import (
    DEFAULT_REPORT_LIMIT,
    MAX_REPORT_LIMIT,
    make_startup_recovery_report_tool,
)


PRIVATE_MARKERS = (
    "/\x55sers/example/private/startup-recovery",
    "/private/startup-recovery",
    "/var/folders/startup-recovery",
    "/tmp/startup-recovery",
    "STARTUP_RECOVERY_PRIVATE_MARKER",
)
STORE_METHODS = (
    "begin_startup_recovery_run",
    "record_startup_recovery_component",
    "heartbeat_startup_recovery_run",
    "finalize_startup_recovery_run",
    "fail_startup_recovery_run",
    "list_startup_recovery_runs",
)
FALSE_BOUNDARIES = (
    "content_exposed",
    "content_in_metadata",
    "reads_database_file",
    "reads_db_file_contents",
    "reads_private_data",
    "reads_personal_data",
    "reads_secret_values",
    "calls_model",
    "calls_external_service",
    "executes_tools",
    "executes_side_effect",
    "external_side_effect",
    "writes_files",
    "edits_files",
    "writes_database",
    "writes_memory",
    "writes_notes",
    "queues_approval",
    "approves_request",
    "dismisses_request",
    "requires_approval",
    "controls_computer",
    "authorizes_execution",
    "authorizes_completion_claim",
    "approval_granted",
    "changed_state",
)


def _expect_error(
    errors: type[BaseException] | tuple[type[BaseException], ...],
    action: Callable[[], object],
    label: str,
) -> None:
    expected = errors if isinstance(errors, tuple) else (errors,)
    try:
        action()
    except expected:
        return
    except BaseException as exc:
        names = ", ".join(item.__name__ for item in expected)
        raise SystemExit(f"{label} raised {type(exc).__name__}, expected {names}") from exc
    names = ", ".join(item.__name__ for item in expected)
    raise SystemExit(f"{label} did not fail closed with {names}")


def _assert_no_private(value: object, label: str) -> None:
    payload = value if isinstance(value, str) else json.dumps(value, sort_keys=True, default=str)
    for marker in PRIVATE_MARKERS:
        if marker in payload:
            raise SystemExit(f"{label} leaked private marker {marker!r}")


def _completed_summary(seed: int = 1) -> dict[str, Any]:
    return {
        "status": "completed",
        "attempted": seed,
        "completed": seed,
        "pending": 0,
    }


def _record_all(
    store: MemoryStore,
    run_id: int,
    *,
    partial_component: str | None = None,
) -> None:
    for index, component in enumerate(STARTUP_RECOVERY_COMPONENTS, start=1):
        summary = _completed_summary(index)
        if component == partial_component:
            summary = {
                "status": "partial",
                "attempted": index,
                "completed": index - 1,
                "pending": 1,
            }
        if not store.record_startup_recovery_component(run_id, component, summary):
            raise SystemExit(f"could not record startup recovery component {component}")


def _state_by_id(store: MemoryStore) -> dict[int, str]:
    return {int(row["run_id"]): str(row["state"]) for row in store.list_startup_recovery_runs(100)}


def test_store_api(store: MemoryStore) -> None:
    completed_run = store.begin_startup_recovery_run("00000001")
    first_component = STARTUP_RECOVERY_COMPONENTS[0]
    if not store.record_startup_recovery_component(
        completed_run,
        first_component,
        _completed_summary(),
    ):
        raise SystemExit("first startup recovery component record should succeed")
    if store.record_startup_recovery_component(
        completed_run,
        first_component,
        _completed_summary(),
    ):
        raise SystemExit("duplicate startup recovery component record should be fenced")
    for index, component in enumerate(STARTUP_RECOVERY_COMPONENTS[1:], start=2):
        if not store.record_startup_recovery_component(
            completed_run,
            component,
            _completed_summary(index),
        ):
            raise SystemExit(f"startup recovery component {component} should record once")
    if store.finalize_startup_recovery_run(completed_run) != "completed":
        raise SystemExit("all-completed startup recovery should finalize completed")
    if store.finalize_startup_recovery_run(completed_run) != "completed":
        raise SystemExit("completed startup recovery finalization should be idempotent")
    if store.fail_startup_recovery_run(completed_run, "RuntimeError"):
        raise SystemExit("completed startup recovery must not transition to failed")
    if store.record_startup_recovery_component(
        completed_run,
        first_component,
        _completed_summary(),
    ):
        raise SystemExit("finalized startup recovery must reject component writes")

    incomplete_run = store.begin_startup_recovery_run("00000002")
    if not store.record_startup_recovery_component(
        incomplete_run,
        first_component,
        _completed_summary(),
    ):
        raise SystemExit("incomplete startup recovery setup failed")
    _expect_error(
        RuntimeError,
        lambda: store.finalize_startup_recovery_run(incomplete_run),
        "incomplete startup recovery finalization",
    )
    if not store.fail_startup_recovery_run(incomplete_run, "RuntimeError"):
        raise SystemExit("incomplete startup recovery should remain fail-able")

    partial_run = store.begin_startup_recovery_run("00000003")
    _record_all(store, partial_run, partial_component="memory")
    if store.finalize_startup_recovery_run(partial_run) != "partial":
        raise SystemExit("mixed startup recovery should finalize partial")

    failed_run = store.begin_startup_recovery_run("00000004")
    if not store.record_startup_recovery_component(
        failed_run,
        "person",
        {"status": "failed", "exception_type": "RuntimeError"},
    ):
        raise SystemExit("failed startup recovery component should be auditable")
    if not store.fail_startup_recovery_run(failed_run, "RuntimeError"):
        raise SystemExit("active startup recovery should transition to failed")
    if not store.fail_startup_recovery_run(failed_run, "RuntimeError"):
        raise SystemExit("identical failed startup recovery replay should be idempotent")
    if store.fail_startup_recovery_run(failed_run, "TimeoutError"):
        raise SystemExit("failed startup recovery must reject contradictory failure replay")
    if store.finalize_startup_recovery_run(failed_run) != "failed":
        raise SystemExit("failed startup recovery finalization should report failed")

    stale_run = store.begin_startup_recovery_run("00000005")
    old_timestamp = "2000-01-01T00:00:00Z"
    with store.connect() as conn:
        conn.execute(
            "UPDATE startup_recovery_runs SET started_at = ?, updated_at = ? WHERE run_id = ?",
            (old_timestamp, old_timestamp, stale_run),
        )
    fresh_run = store.begin_startup_recovery_run("00000006")
    states = _state_by_id(store)
    if states.get(stale_run) != "interrupted" or states.get(fresh_run) != "running":
        raise SystemExit(f"stale/fresh startup recovery fencing diverged: {states}")
    concurrent_fresh_run = store.begin_startup_recovery_run("00000007")
    states = _state_by_id(store)
    if states.get(fresh_run) != "running" or states.get(concurrent_fresh_run) != "running":
        raise SystemExit(f"fresh concurrent startup recovery was incorrectly interrupted: {states}")
    if not store.fail_startup_recovery_run(fresh_run, "RuntimeError"):
        raise SystemExit("fresh concurrent run cleanup failed")
    if not store.fail_startup_recovery_run(concurrent_fresh_run, "RuntimeError"):
        raise SystemExit("newest concurrent run cleanup failed")

    heartbeat_run = store.begin_startup_recovery_run("0000000a")
    with store.connect() as conn:
        conn.execute(
            "UPDATE startup_recovery_runs SET started_at = ?, updated_at = ? WHERE run_id = ?",
            (old_timestamp, old_timestamp, heartbeat_run),
        )
    if not store.heartbeat_startup_recovery_run(heartbeat_run):
        raise SystemExit("active startup recovery heartbeat should renew its lease")
    post_heartbeat_run = store.begin_startup_recovery_run("0000000b")
    states = _state_by_id(store)
    if states.get(heartbeat_run) != "running":
        raise SystemExit(f"renewed startup recovery lease was interrupted: {states}")
    if not store.fail_startup_recovery_run(heartbeat_run, "RuntimeError"):
        raise SystemExit("heartbeat run cleanup failed")
    if not store.fail_startup_recovery_run(post_heartbeat_run, "RuntimeError"):
        raise SystemExit("post-heartbeat run cleanup failed")
    if store.heartbeat_startup_recovery_run(heartbeat_run):
        raise SystemExit("terminal startup recovery heartbeat should be fenced")

    rollback_run = store.begin_startup_recovery_run("0000000c")
    future_timestamp = "2099-01-01T00:00:00Z"
    with store.connect() as conn:
        conn.execute(
            "UPDATE startup_recovery_runs SET started_at = ?, updated_at = ? WHERE run_id = ?",
            (future_timestamp, future_timestamp, rollback_run),
        )
    if not store.heartbeat_startup_recovery_run(rollback_run):
        raise SystemExit("clock-rollback startup recovery heartbeat should clamp forward")
    if not store.record_startup_recovery_component(
        rollback_run,
        "person",
        _completed_summary(),
    ):
        raise SystemExit("clock-rollback startup recovery component should clamp forward")
    if not store.fail_startup_recovery_run(rollback_run, "RuntimeError"):
        raise SystemExit("clock-rollback startup recovery failure should clamp forward")
    rollback_row = next(
        row
        for row in store.list_startup_recovery_runs(100)
        if int(row["run_id"]) == rollback_run
    )
    if (
        rollback_row["updated_at"] != future_timestamp
        or rollback_row["finished_at"] != future_timestamp
    ):
        raise SystemExit(f"clock-rollback receipt timestamps regressed: {dict(rollback_row)}")
    with store.connect() as conn:
        conn.execute("DELETE FROM startup_recovery_runs WHERE run_id = ?", (rollback_run,))

    future_crash_run = store.begin_startup_recovery_run("0000000d")
    with store.connect() as conn:
        conn.execute(
            "UPDATE startup_recovery_runs SET started_at = ?, updated_at = ? WHERE run_id = ?",
            (future_timestamp, future_timestamp, future_crash_run),
        )
    post_rollback_run = store.begin_startup_recovery_run("0000000e")
    states = _state_by_id(store)
    if (
        states.get(future_crash_run) != "interrupted"
        or states.get(post_rollback_run) != "running"
        or int(store.list_startup_recovery_runs(1)[0]["run_id"]) != post_rollback_run
    ):
        raise SystemExit(f"future-dated crashed startup recovery was not fenced: {states}")
    if not store.fail_startup_recovery_run(post_rollback_run, "RuntimeError"):
        raise SystemExit("post-rollback run cleanup failed")

    private = "/\x55sers/example/private/startup-recovery/store.sqlite"
    for bad_session in ("", "smoke", "1234567g", private, True, None, "x" * 8):
        _expect_error(
            (TypeError, ValueError),
            lambda value=bad_session: store.begin_startup_recovery_run(value),  # type: ignore[arg-type]
            "malformed startup recovery session id",
        )
    malformed_summaries: tuple[object, ...] = (
        None,
        [],
        {},
        {"status": "running"},
        {"status": "completed", "attempted": True},
        {"status": "completed", "attempted": -1},
        {"status": "completed", "pending": 1},
        {"status": "completed", "attempted": 0, "completed": 1, "pending": 0},
        {
            "status": "completed",
            "attempted": 1,
            "completed": 1,
            "pending": 0,
            "custody_conflicts": 1,
        },
        {"status": "partial", "attempted": 1, "completed": 2, "pending": 1},
        {"status": "completed", "exception_type": "UnexpectedError"},
        {"status": "partial", "pending": 0},
        {"status": "failed"},
        {"status": "completed", "private": private},
        {"status": "failed", "exception_type": private},
    )
    validation_run = store.begin_startup_recovery_run("00000008")
    for summary in malformed_summaries:
        _expect_error(
            (TypeError, ValueError),
            lambda value=summary: store.record_startup_recovery_component(
                validation_run,
                "person",
                value,  # type: ignore[arg-type]
            ),
            "malformed startup recovery summary",
        )
    conflict_summary = {
        "status": "partial",
        "attempted": 0,
        "completed": 0,
        "pending": 3,
        "custody_backfilled": 20,
        "custody_conflicts": 3,
    }
    if not store.record_startup_recovery_component(
        validation_run,
        "preference",
        conflict_summary,
    ):
        raise SystemExit("valid preference custody recovery evidence was rejected")
    for bad_component in ("", "contacts", private, True, None):
        _expect_error(
            (TypeError, ValueError),
            lambda value=bad_component: store.record_startup_recovery_component(
                validation_run,
                value,  # type: ignore[arg-type]
                _completed_summary(),
            ),
            "malformed startup recovery component",
        )
    for bad_run_id in (0, -1, True, "1", None):
        _expect_error(
            (TypeError, ValueError),
            lambda value=bad_run_id: store.finalize_startup_recovery_run(value),  # type: ignore[arg-type]
            "malformed startup recovery run id",
        )
    _expect_error(
        ValueError,
        lambda: store.finalize_startup_recovery_run(9_223_372_036_854_775_807),
        "missing startup recovery run",
    )
    for bad_exception in ("", "CustomError", private, True, None, "x" * 129):
        _expect_error(
            (TypeError, ValueError),
            lambda value=bad_exception: store.fail_startup_recovery_run(
                validation_run,
                value,  # type: ignore[arg-type]
            ),
            "malformed startup recovery exception type",
        )
    for bad_limit in (0, -1, 101, True, "10", None):
        _expect_error(
            ValueError,
            lambda value=bad_limit: store.list_startup_recovery_runs(value),  # type: ignore[arg-type]
            "malformed startup recovery list limit",
        )
    if not store.fail_startup_recovery_run(validation_run, "RuntimeError"):
        raise SystemExit("validation startup recovery cleanup failed")

    completed_details = next(
        str(row["details"])
        for row in store.list_startup_recovery_runs(100)
        if int(row["run_id"]) == completed_run
    )
    with store.connect() as conn:
        conn.execute(
            "UPDATE startup_recovery_runs SET details = '{}' WHERE run_id = ?",
            (completed_run,),
        )
    _expect_error(
        RuntimeError,
        lambda: store.finalize_startup_recovery_run(completed_run),
        "malformed terminal startup recovery finalization",
    )
    _expect_error(
        RuntimeError,
        lambda: store.list_startup_recovery_runs(100),
        "malformed terminal startup recovery listing",
    )
    with store.connect() as conn:
        conn.execute(
            "UPDATE startup_recovery_runs SET details = ? WHERE run_id = ?",
            (completed_details, completed_run),
        )

    bounded_rows = store.list_startup_recovery_runs(3)
    if len(bounded_rows) != 3:
        raise SystemExit(f"startup recovery list limit was not exact: {len(bounded_rows)}")
    all_rows = store.list_startup_recovery_runs(100)
    if any(
        not {"run_id", "state", "details"}.issubset(set(row.keys()))
        for row in all_rows
    ):
        raise SystemExit("startup recovery list rows missed required ledger fields")
    _assert_no_private([dict(row) for row in all_rows], "startup recovery store rows")


def _canonical_details(
    *,
    component_status: str = "completed",
    all_components: bool = True,
) -> str:
    components = STARTUP_RECOVERY_COMPONENTS if all_components else STARTUP_RECOVERY_COMPONENTS[:1]
    payload = {
        component: {
            "attempted": index,
            "completed": index if component_status == "completed" else max(index - 1, 0),
            "pending": 0 if component_status == "completed" else 1,
            "status": component_status,
        }
        for index, component in enumerate(components, start=1)
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


class FakeStore:
    def __init__(self, rows: list[Any] | None = None, error: BaseException | None = None):
        self.rows = rows or []
        self.error = error
        self.limits: list[int] = []

    def list_startup_recovery_runs(self, limit: int) -> list[Any]:
        self.limits.append(limit)
        if self.error is not None:
            raise self.error
        return self.rows[:limit]


def _assert_boundaries(metadata: dict[str, Any], label: str) -> None:
    if metadata.get("metadata_only") is not True or metadata.get("content_free") is not True:
        raise SystemExit(f"{label} missed content-free metadata boundary: {metadata}")
    for key in FALSE_BOUNDARIES:
        if metadata.get(key) is not False:
            raise SystemExit(f"{label} boundary {key} should be exact false: {metadata}")
    handoff = metadata.get("startup_recovery_handoff")
    if not isinstance(handoff, dict):
        raise SystemExit(f"{label} missed structured startup recovery handoff: {metadata}")
    if (
        handoff.get("source") != "startup_recovery_report"
        or handoff.get("handoff_ready") is not True
        or handoff.get("startup_recovery_handoff_ready") is not True
        or handoff.get("review_only") is not True
        or handoff.get("metadata_only") is not True
        or handoff.get("content_free") is not True
        or handoff.get("content_in_handoff") is not False
        or handoff.get("loads_without_execution") is not True
    ):
        raise SystemExit(f"{label} handoff contract diverged: {handoff}")
    for key in (
        "calls_model",
        "executes_tools",
        "writes_files",
        "writes_database",
        "writes_memory",
        "writes_notes",
        "reads_private_data",
        "reads_personal_data",
        "external_side_effect",
        "controls_computer",
        "queues_approval",
        "authorizes_execution",
        "authorizes_completion_claim",
        "approval_granted",
    ):
        if handoff.get(key) is not False:
            raise SystemExit(f"{label} handoff {key} should be exact false: {handoff}")


def test_report_parser_and_boundaries(store: MemoryStore) -> None:
    report_run = store.begin_startup_recovery_run("00000009")
    _record_all(store, report_run)
    if store.finalize_startup_recovery_run(report_run) != "completed":
        raise SystemExit("report fixture should finalize completed")
    report = make_startup_recovery_report_tool(store)({"limit": 3})
    if not report.ok or report.tool_name != "startup_recovery_report":
        raise SystemExit(f"real-store startup recovery report failed: {report}")
    if report.metadata.get("limit") != 3 or report.metadata.get("rows_reviewed") != 3:
        raise SystemExit(f"startup recovery report did not honor its bound: {report.metadata}")
    latest = report.metadata.get("latest") or {}
    if latest.get("status") != "completed" or latest.get("component_count") != 5:
        raise SystemExit(f"startup recovery report missed latest completed run: {report.metadata}")
    if latest.get("component_status_counts") != {"completed": 5}:
        raise SystemExit(f"startup recovery report component counts diverged: {latest}")
    if latest.get("recovery_counts") != {"attempted": 15, "completed": 15}:
        raise SystemExit(f"startup recovery report aggregate counts diverged: {latest}")
    if report.metadata.get("history_count") != 2:
        raise SystemExit(f"startup recovery report history was not bounded: {report.metadata}")
    _assert_boundaries(report.metadata, "real startup recovery report")
    _assert_no_private(report.output + json.dumps(report.metadata, sort_keys=True), "real report")

    marker = "STARTUP_RECOVERY_PRIVATE_MARKER /\x55sers/example/private/startup-recovery/audit.json"
    hostile_rows = [
        {
            "run_id": 99,
            "state": marker,
            "details": marker,
        },
        {
            "run_id": 98,
            "state": "completed",
            "details": _canonical_details(),
        },
        {
            "run_id": 97,
            "state": "failed",
            "details": json.dumps(
                {"person": {"status": "failed", "private": marker}},
                sort_keys=True,
                separators=(",", ":"),
            ),
        },
    ]
    hostile_store = FakeStore(hostile_rows)
    hostile = make_startup_recovery_report_tool(hostile_store)({"limit": 20})  # type: ignore[arg-type]
    if not hostile.ok:
        raise SystemExit(f"startup recovery report should hide malformed rows: {hostile}")
    if hostile.metadata.get("latest") is not None or hostile.metadata.get("latest_unreadable") is not True:
        raise SystemExit(f"malformed latest row was silently replaced: {hostile.metadata}")
    if hostile.metadata.get("readable_run_rows") != 1 or hostile.metadata.get("unreadable_run_rows") != 2:
        raise SystemExit(f"malformed startup recovery row counts diverged: {hostile.metadata}")
    if hostile.metadata.get("history_count") != 1:
        raise SystemExit(f"readable startup recovery history was not preserved: {hostile.metadata}")
    _assert_boundaries(hostile.metadata, "hostile startup recovery report")
    _assert_no_private(
        hostile.output + json.dumps(hostile.metadata, sort_keys=True, default=str),
        "hostile startup recovery report",
    )

    inconsistent_rows = [
        {
            "run_id": 4,
            "state": "completed",
            "details": _canonical_details(all_components=False),
        },
        {
            "run_id": 3,
            "state": "completed",
            "details": _canonical_details(component_status="partial"),
        },
        {
            "run_id": 2,
            "state": "partial",
            "details": _canonical_details(),
        },
        {
            "run_id": 1,
            "state": "running",
            "details": '{"person": {"status": "completed"}}',
        },
    ]
    inconsistent = make_startup_recovery_report_tool(FakeStore(inconsistent_rows))({})  # type: ignore[arg-type]
    if inconsistent.metadata.get("readable_run_rows") != 0 or inconsistent.metadata.get("unreadable_run_rows") != 4:
        raise SystemExit(f"inconsistent startup recovery rows should fail closed: {inconsistent.metadata}")
    _assert_no_private(inconsistent.output + repr(inconsistent.metadata), "inconsistent report")

    contradictory_rows = [
        {
            "run_id": 7,
            "state": "completed",
            "details": json.dumps(
                {
                    component: {
                        "status": "completed",
                        "attempted": 0,
                        "completed": 1,
                        "pending": 0,
                    }
                    for component in STARTUP_RECOVERY_COMPONENTS
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
        },
        {
            "run_id": 6,
            "state": "failed",
            "details": json.dumps(
                {
                    "person": {
                        "status": "failed",
                        "exception_type": "RuntimeError",
                        "attempted": 1,
                    }
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
        },
    ]
    contradictory = make_startup_recovery_report_tool(FakeStore(contradictory_rows))({})  # type: ignore[arg-type]
    if (
        contradictory.metadata.get("readable_run_rows") != 0
        or contradictory.metadata.get("unreadable_run_rows") != 2
    ):
        raise SystemExit(
            f"contradictory startup recovery counts should fail closed: {contradictory.metadata}"
        )

    list_details = [
        {"component": component, **_completed_summary(index)}
        for index, component in enumerate(STARTUP_RECOVERY_COMPONENTS, start=1)
    ]
    list_report = make_startup_recovery_report_tool(
        FakeStore([{"run_id": 5, "state": "completed", "details": list_details}])
    )({})  # type: ignore[arg-type]
    list_latest = list_report.metadata.get("latest") or {}
    if (
        not list_report.ok
        or list_latest.get("component_status_counts") != {"completed": 5}
        or list_latest.get("recovery_counts") != {"attempted": 15, "completed": 15}
    ):
        raise SystemExit(f"list-shaped startup recovery compatibility diverged: {list_report}")

    limit_store = FakeStore(hostile_rows * 10)
    limit_report = make_startup_recovery_report_tool(limit_store)
    limit_cases = (
        ({"limit": True}, DEFAULT_REPORT_LIMIT),
        ({"limit": "20"}, DEFAULT_REPORT_LIMIT),
        ({"limit": -10}, 1),
        ({"limit": 10_000}, MAX_REPORT_LIMIT),
        (marker, DEFAULT_REPORT_LIMIT),
    )
    for args, expected in limit_cases:
        result = limit_report(args)  # type: ignore[arg-type]
        if result.metadata.get("limit") != expected or limit_store.limits[-1] != expected:
            raise SystemExit(f"startup recovery report limit boundary diverged: {args!r} -> {result.metadata}")
        if result.metadata.get("rows_reviewed", 0) > expected:
            raise SystemExit(f"startup recovery report exceeded row bound: {result.metadata}")

    unavailable = make_startup_recovery_report_tool(
        FakeStore(error=RuntimeError(marker))  # type: ignore[arg-type]
    )({"limit": 2})
    if unavailable.ok or unavailable.metadata.get("report_status") != "unavailable":
        raise SystemExit(f"unavailable startup recovery audit should fail closed: {unavailable}")
    for expected in (
        "storage status",
        "setup check",
        "repair the reported storage access",
        "startup recovery report",
    ):
        if expected not in unavailable.output:
            raise SystemExit(
                f"unavailable startup recovery audit missed recovery text {expected!r}: "
                f"{unavailable.output}"
            )
    if unavailable.metadata.get("next_command") != "storage status":
        raise SystemExit(
            f"unavailable startup recovery audit missed next command: {unavailable.metadata}"
        )
    if unavailable.metadata.get("recovery_commands") != [
        "storage status",
        "setup check",
        "startup recovery report",
    ]:
        raise SystemExit(
            "unavailable startup recovery audit missed bounded recovery commands: "
            f"{unavailable.metadata}"
        )
    if unavailable.metadata.get("retry_requires_storage_repair") is not True:
        raise SystemExit(
            f"unavailable startup recovery audit should require storage repair: {unavailable.metadata}"
        )
    _assert_boundaries(unavailable.metadata, "unavailable startup recovery report")
    _assert_no_private(
        unavailable.output + json.dumps(unavailable.metadata, sort_keys=True),
        "unavailable startup recovery report",
    )


def test_planner_and_optional_runtime_integration(root: Path, store: MemoryStore) -> None:
    for command in (
        "startup recovery report",
        "did startup recovery run?",
        "what did jarvis repair at startup?",
    ):
        plan = RuleBasedPlanner().plan(command)
        if len(plan.actions) != 1 or plan.actions[0].tool_name != "startup_recovery_report":
            raise SystemExit(f"startup recovery planner route diverged for {command!r}: {plan}")
    print("[ok] startup recovery planner route")

    vault = ObsidianVault(root / "Vault", "Jarvis")
    vault.init()
    config = JarvisConfig(
        data_dir=root,
        db_path=store.db_path,
        obsidian_vault=root / "Vault",
        obsidian_root="Jarvis",
        use_model_planner=False,
    )
    registry = build_core_registry(store, vault, config, "startup-smoke")
    try:
        registered = registry.get("startup_recovery_report")
    except KeyError as exc:
        raise SystemExit("startup_recovery_report is missing from the core registry") from exc
    if registered.risk.name != "READ_ONLY":
        raise SystemExit(f"startup recovery report registry risk diverged: {registered.risk}")
    contract = registered.argument_contract
    if (
        contract is None
        or contract.version != 1
        or contract.allow_unknown
        or len(contract.fields) != 1
        or contract.fields[0].name != "limit"
        or contract.fields[0].required
        or contract.fields[0].types != frozenset({ToolArgumentType.INTEGER})
    ):
        raise SystemExit(f"startup recovery report argument contract diverged: {contract}")

    from jarvis_v2.agent.runtime import JarvisRuntime

    runtime = JarvisRuntime(config)
    latest_runtime_row = runtime.store.list_startup_recovery_runs(1)[0]
    latest_runtime_details = json.loads(str(latest_runtime_row["details"]))
    if (
        runtime.startup_recovery_audit_status.get("status") != "completed"
        or latest_runtime_row["state"] != "completed"
        or set(latest_runtime_details) != set(STARTUP_RECOVERY_COMPONENTS)
    ):
        raise SystemExit(
            f"runtime startup recovery receipt did not close all components: "
            f"{runtime.startup_recovery_audit_status} / {dict(latest_runtime_row)}"
        )
    result = runtime.handle("startup recovery report")
    if (
        not result.verified
        or len(result.tool_results) != 1
        or result.tool_results[0].tool_name != "startup_recovery_report"
        or not result.tool_results[0].ok
    ):
        raise SystemExit(f"startup recovery runtime integration diverged: {result}")
    _assert_boundaries(result.tool_results[0].metadata, "runtime startup recovery report")
    print("[ok] startup recovery registry/runtime integration")

    from jarvis_v2.agent import runtime as runtime_module

    if (
        runtime_module.STARTUP_RECOVERY_HEARTBEAT_JOIN_SECONDS
        <= SQLITE_BUSY_TIMEOUT_SECONDS
    ):
        raise SystemExit(
            "startup recovery heartbeat shutdown bound must exceed SQLite's lock wait"
        )

    partial_root = root / "partial-runtime"
    partial_config = JarvisConfig(
        data_dir=partial_root,
        db_path=partial_root / "jarvis.sqlite",
        obsidian_vault=partial_root / "Vault",
        obsidian_root="Jarvis",
        use_model_planner=False,
    )
    with patch.object(
        runtime_module,
        "reconcile_pending_memory_projections",
        return_value=SimpleNamespace(attempted=20, completed=20, pending=1),
    ):
        partial_runtime = JarvisRuntime(partial_config)
    partial_row = partial_runtime.store.list_startup_recovery_runs(1)[0]
    partial_details = json.loads(str(partial_row["details"]))
    if (
        partial_runtime.memory_projection_recovery_status.get("status") != "pending"
        or partial_runtime.startup_recovery_audit_status.get("status") != "partial"
        or partial_row["state"] != "partial"
        or partial_details.get("memory", {}).get("status") != "partial"
        or partial_details.get("memory", {}).get("pending") != 1
    ):
        raise SystemExit(
            f"runtime pending recovery was not durably normalized to partial: "
            f"{partial_runtime.startup_recovery_audit_status} / {dict(partial_row)}"
        )

    failure_marker = (
        "STARTUP_RECOVERY_PRIVATE_MARKER /\x55sers/example/private/startup-recovery/failure.txt"
    )
    failed_component_root = root / "failed-component-runtime"
    failed_component_config = JarvisConfig(
        data_dir=failed_component_root,
        db_path=failed_component_root / "jarvis.sqlite",
        obsidian_vault=failed_component_root / "Vault",
        obsidian_root="Jarvis",
        use_model_planner=False,
    )
    with patch.object(
        runtime_module,
        "reconcile_pending_person_projections",
        side_effect=RuntimeError(failure_marker),
    ):
        failed_component_runtime = JarvisRuntime(failed_component_config)
    failed_component_row = failed_component_runtime.store.list_startup_recovery_runs(1)[0]
    failed_component_details = json.loads(str(failed_component_row["details"]))
    failed_component_payload = json.dumps(
        dict(failed_component_row), sort_keys=True, default=str
    )
    if (
        failed_component_runtime.startup_recovery_audit_status.get("status") != "partial"
        or failed_component_row["state"] != "partial"
        or set(failed_component_details) != set(STARTUP_RECOVERY_COMPONENTS)
        or failed_component_details.get("person", {})
        .get("exception_type")
        != "RuntimeError"
        or failed_component_details.get("skill", {}).get("status") != "completed"
        or failure_marker in failed_component_payload
    ):
        raise SystemExit(
            f"runtime component failure receipt was not content-free partial: "
            f"{failed_component_runtime.startup_recovery_audit_status} / "
            f"{failed_component_payload}"
        )

    begin_failure_root = root / "begin-failure-runtime"
    begin_failure_config = JarvisConfig(
        data_dir=begin_failure_root,
        db_path=begin_failure_root / "jarvis.sqlite",
        obsidian_vault=begin_failure_root / "Vault",
        obsidian_root="Jarvis",
        use_model_planner=False,
    )
    with (
        patch.object(
            MemoryStore,
            "begin_startup_recovery_run",
            side_effect=RuntimeError(failure_marker),
        ),
        patch.object(ObsidianVault, "init") as forbidden_vault,
        patch.object(runtime_module, "reconcile_pending_person_projections") as forbidden_repair,
    ):
        try:
            JarvisRuntime(begin_failure_config)
        except StartupRecoveryUnavailable as exc:
            rendered = "".join(traceback.format_exception(exc))
            if failure_marker in rendered:
                raise SystemExit("startup audit begin failure exposed private traceback detail")
        else:
            raise SystemExit("runtime should stop when startup recovery audit cannot begin")
        if forbidden_vault.called:
            raise SystemExit("runtime initialized the vault before startup audit custody existed")
        if forbidden_repair.called:
            raise SystemExit("runtime repaired projections after startup audit begin failed")

    thread_start_root = root / "thread-start-failure-runtime"
    thread_start_config = JarvisConfig(
        data_dir=thread_start_root,
        db_path=thread_start_root / "jarvis.sqlite",
        obsidian_vault=thread_start_root / "Vault",
        obsidian_root="Jarvis",
        use_model_planner=False,
    )
    with patch.object(
        runtime_module.threading.Thread,
        "start",
        side_effect=RuntimeError(failure_marker),
    ):
        try:
            JarvisRuntime(thread_start_config)
        except StartupRecoveryUnavailable as exc:
            rendered = "".join(traceback.format_exception(exc))
            if failure_marker in rendered:
                raise SystemExit("heartbeat thread-start failure exposed private traceback detail")
        else:
            raise SystemExit("runtime should stop when startup audit heartbeat cannot start")
    thread_start_row = MemoryStore(thread_start_config.db_path).list_startup_recovery_runs(1)[0]
    if thread_start_row["state"] != "failed":
        raise SystemExit(f"heartbeat thread-start failure was not durably failed: {dict(thread_start_row)}")

    late_heartbeat_root = root / "late-heartbeat-runtime"
    late_heartbeat_config = JarvisConfig(
        data_dir=late_heartbeat_root,
        db_path=late_heartbeat_root / "jarvis.sqlite",
        obsidian_vault=late_heartbeat_root / "Vault",
        obsidian_root="Jarvis",
        use_model_planner=False,
    )
    real_heartbeat = MemoryStore.heartbeat_startup_recovery_run
    real_person_reconcile = runtime_module.reconcile_pending_person_projections
    heartbeat_calls = 0

    def _lose_second_heartbeat(store: MemoryStore, run_id: int) -> bool:
        nonlocal heartbeat_calls
        heartbeat_calls += 1
        if heartbeat_calls >= 2:
            return False
        return real_heartbeat(store, run_id)

    def _slow_person_reconcile(*args: Any, **kwargs: Any) -> Any:
        time.sleep(0.04)
        return real_person_reconcile(*args, **kwargs)

    with (
        patch.object(runtime_module, "STARTUP_RECOVERY_HEARTBEAT_SECONDS", 0.01),
        patch.object(
            MemoryStore,
            "heartbeat_startup_recovery_run",
            autospec=True,
            side_effect=_lose_second_heartbeat,
        ),
        patch.object(
            runtime_module,
            "reconcile_pending_person_projections",
            side_effect=_slow_person_reconcile,
        ),
    ):
        try:
            JarvisRuntime(late_heartbeat_config)
        except StartupRecoveryUnavailable:
            pass
        else:
            raise SystemExit("runtime should stop after a late startup audit heartbeat loss")
    late_heartbeat_row = MemoryStore(late_heartbeat_config.db_path).list_startup_recovery_runs(1)[0]
    if late_heartbeat_row["state"] != "failed" or heartbeat_calls < 2:
        raise SystemExit(
            f"late heartbeat loss was not durably failed: {dict(late_heartbeat_row)}"
        )

    blocked_heartbeat_root = root / "blocked-heartbeat-runtime"
    blocked_heartbeat_config = JarvisConfig(
        data_dir=blocked_heartbeat_root,
        db_path=blocked_heartbeat_root / "jarvis.sqlite",
        obsidian_vault=blocked_heartbeat_root / "Vault",
        obsidian_root="Jarvis",
        use_model_planner=False,
    )

    def _slow_heartbeat(store: MemoryStore, run_id: int) -> bool:
        time.sleep(1.05)
        return real_heartbeat(store, run_id)

    with patch.object(
        MemoryStore,
        "heartbeat_startup_recovery_run",
        autospec=True,
        side_effect=_slow_heartbeat,
    ):
        blocked_heartbeat_runtime = JarvisRuntime(blocked_heartbeat_config)
    blocked_heartbeat_row = blocked_heartbeat_runtime.store.list_startup_recovery_runs(1)[0]
    if blocked_heartbeat_row["state"] != "completed":
        raise SystemExit(
            "routine heartbeat lock-wait window incorrectly failed startup: "
            f"{dict(blocked_heartbeat_row)}"
        )

    record_failure_root = root / "record-failure-runtime"
    record_failure_config = JarvisConfig(
        data_dir=record_failure_root,
        db_path=record_failure_root / "jarvis.sqlite",
        obsidian_vault=record_failure_root / "Vault",
        obsidian_root="Jarvis",
        use_model_planner=False,
    )
    with patch.object(
        MemoryStore,
        "record_startup_recovery_component",
        side_effect=RuntimeError(failure_marker),
    ):
        try:
            JarvisRuntime(record_failure_config)
        except StartupRecoveryUnavailable as exc:
            rendered = "".join(traceback.format_exception(exc))
            if failure_marker in rendered:
                raise SystemExit("startup audit record failure exposed private traceback detail")
        else:
            raise SystemExit("runtime should stop after startup audit component custody loss")
    record_failure_store = MemoryStore(record_failure_config.db_path)
    record_failure_row = record_failure_store.list_startup_recovery_runs(1)[0]
    if (
        record_failure_row["state"] != "failed"
        or record_failure_row["exception_type"] != "RuntimeError"
    ):
        raise SystemExit(
            f"startup audit component custody loss was not durably failed: "
            f"{dict(record_failure_row)}"
        )

    finalize_failure_root = root / "finalize-failure-runtime"
    finalize_failure_config = JarvisConfig(
        data_dir=finalize_failure_root,
        db_path=finalize_failure_root / "jarvis.sqlite",
        obsidian_vault=finalize_failure_root / "Vault",
        obsidian_root="Jarvis",
        use_model_planner=False,
    )
    with patch.object(
        MemoryStore,
        "finalize_startup_recovery_run",
        side_effect=RuntimeError(failure_marker),
    ):
        try:
            JarvisRuntime(finalize_failure_config)
        except StartupRecoveryUnavailable as exc:
            rendered = "".join(traceback.format_exception(exc))
            if failure_marker in rendered:
                raise SystemExit("startup audit finalize failure exposed private traceback detail")
        else:
            raise SystemExit("runtime should stop when startup audit finalization fails")
    finalize_failure_row = MemoryStore(finalize_failure_config.db_path).list_startup_recovery_runs(1)[0]
    if (
        finalize_failure_row["state"] != "failed"
        or finalize_failure_row["exception_type"] != "RuntimeError"
    ):
        raise SystemExit(
            "startup audit finalize failure was not durably failed: "
            f"{dict(finalize_failure_row)}"
        )

    read_only_root = root / "read-only-runtime"
    read_only_config = JarvisConfig(
        data_dir=read_only_root,
        db_path=read_only_root / "jarvis.sqlite",
        obsidian_vault=read_only_root / "Vault",
        obsidian_root="Jarvis",
        use_model_planner=False,
    )

    def _read_only_init(store: MemoryStore) -> None:
        store.read_only_startup = True

    with (
        patch.object(MemoryStore, "init", _read_only_init),
        patch.object(MemoryStore, "begin_startup_recovery_run") as forbidden_begin,
        patch.object(ObsidianVault, "init") as forbidden_vault,
        patch.object(runtime_module, "reconcile_pending_person_projections") as forbidden_repair,
    ):
        read_only_runtime = JarvisRuntime(read_only_config)
    if (
        read_only_runtime.startup_recovery_audit_status.get("status")
        != "skipped_read_only_startup"
        or forbidden_begin.called
        or forbidden_vault.called
        or forbidden_repair.called
    ):
        raise SystemExit(
            "read-only startup did not skip startup recovery without writes: "
            f"{read_only_runtime.startup_recovery_audit_status}"
        )
    print("[ok] startup recovery runtime fail-closed and partial receipts")


def main() -> None:
    missing = [name for name in STORE_METHODS if not callable(getattr(MemoryStore, name, None))]
    if missing:
        raise SystemExit(
            f"startup recovery MemoryStore APIs are not wired: {', '.join(missing)}"
        )

    with TemporaryDirectory(prefix="jarvis-startup-recovery-audit-") as temp:
        root = Path(temp)
        incompatible_path = root / "incompatible.sqlite"
        with sqlite3.connect(incompatible_path) as conn:
            conn.execute("CREATE TABLE startup_recovery_runs(run_id INTEGER PRIMARY KEY)")
        incompatible_store = MemoryStore(incompatible_path)
        _expect_error(
            RuntimeError,
            incompatible_store.init,
            "incompatible startup recovery ledger schema",
        )
        weak_schema_path = root / "weak-schema.sqlite"
        with sqlite3.connect(weak_schema_path) as conn:
            conn.execute(
                """
                CREATE TABLE startup_recovery_runs (
                    run_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    runtime_session_id TEXT NOT NULL,
                    state TEXT NOT NULL DEFAULT 'running',
                    details TEXT NOT NULL DEFAULT '{}',
                    exception_type TEXT,
                    revision INTEGER NOT NULL DEFAULT 0,
                    started_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    finished_at TEXT
                )
                """
            )
        weak_schema_store = MemoryStore(weak_schema_path)
        _expect_error(
            RuntimeError,
            weak_schema_store.init,
            "same-shaped weak startup recovery ledger schema",
        )
        case_schema_path = root / "case-schema.sqlite"
        with sqlite3.connect(case_schema_path) as conn:
            conn.execute(
                STARTUP_RECOVERY_TABLE_SQL.replace(
                    "'RuntimeError'",
                    "'runtimeerror'",
                )
            )
        case_schema_store = MemoryStore(case_schema_path)
        _expect_error(
            RuntimeError,
            case_schema_store.init,
            "case-drifted startup recovery ledger schema",
        )
        store = MemoryStore(root / "jarvis.sqlite")
        store.init()
        test_store_api(store)
        print("[ok] startup recovery store state machine")
        test_report_parser_and_boundaries(store)
        print("[ok] startup recovery report metadata, bounds, and privacy")
        test_planner_and_optional_runtime_integration(root, store)
    print(
        "Startup recovery audit smoke passed "
        f"(stale threshold {STARTUP_RECOVERY_STALE_AFTER_SECONDS}s)"
    )


if __name__ == "__main__":
    main()
