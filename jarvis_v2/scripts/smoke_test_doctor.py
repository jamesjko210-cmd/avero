"""Smoke tests for Jarvis doctor readiness gates (mocked diagnostics, no writes)."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from jarvis_v2.agent.planner import RuleBasedPlanner
from jarvis_v2.config import JarvisConfig
from jarvis_v2.scripts.test_runtime import make_temp_runtime
from jarvis_v2.tools import doctor as dt
from jarvis_v2.tools import storage as st


def assert_doctor_metadata_bool_is_exact() -> None:
    if dt._metadata_bool(True) is not True:
        raise SystemExit("doctor exact metadata bool rejected True")
    if dt._metadata_bool(False) is not False:
        raise SystemExit("doctor exact metadata bool rejected False")
    for value in ("true", "false", "yes", "no", 1, 0, [True], {"ready": True}, None):
        if dt._metadata_bool(value) is not False:
            raise SystemExit(f"doctor exact metadata bool accepted malformed truthy value: {value!r}")
    if dt._metadata_bool("false", default=True) is not True:
        raise SystemExit("doctor exact metadata bool did not preserve explicit default")


def _malformed_ready_diagnostics() -> dict[str, Any]:
    return {
        "available": "true",
        "status": "ready",
        "data_dir": "workspace storage data",
        "db_path": "workspace storage db",
        "db_parent": "workspace storage",
        "db_exists": "true",
        "data_dir_exists": "true",
        "db_parent_exists": "true",
        "data_dir_writable": "true",
        "db_parent_writable": "true",
        "db_file_writable": "true",
        "obsidian_vault": "workspace notes",
        "obsidian_root": "Jarvis",
        "obsidian_root_path": "workspace notes root",
        "obsidian_vault_exists": "true",
        "obsidian_root_exists": "true",
        "obsidian_vault_writable": "true",
        "obsidian_root_writable": "true",
        "workspace_local_notes": "true",
        "metadata_only": "true",
        "issues": ["diagnostic booleans were malformed"],
        "recovery_check_command": st.BOOTSTRAP_CHECK_COMMAND,
        "recovery_check_api": st.STORAGE_RECOVERY_CHECK_API,
        "recovery_command": st.BOOTSTRAP_WRITE_COMMAND,
    }


def _assert_malformed_diagnostic_flags_fail_closed(metadata: dict[str, Any]) -> None:
    for key in [
        "storage_configured_available",
        "storage_db_exists",
        "storage_data_dir_exists",
        "storage_db_parent_exists",
        "storage_data_dir_writable",
        "storage_db_parent_writable",
        "storage_db_file_writable",
        "storage_obsidian_vault_exists",
        "storage_obsidian_root_exists",
        "storage_obsidian_vault_writable",
        "storage_obsidian_root_writable",
        "storage_workspace_local_notes",
        "storage_metadata_only",
    ]:
        if metadata.get(key) is not False:
            raise SystemExit(f"jarvis doctor accepted malformed diagnostic boolean for {key}: {metadata}")


def test_doctor_rejects_malformed_storage_diagnostic_bools() -> None:
    original_configured_storage_diagnostics = dt._configured_storage_diagnostics

    def fake_configured_storage_diagnostics(
        config: JarvisConfig,
        runtime_storage_fallback: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], str]:
        return _malformed_ready_diagnostics(), "mocked"

    with TemporaryDirectory() as tmp:
        runtime = make_temp_runtime(Path(tmp))
        try:
            dt._configured_storage_diagnostics = fake_configured_storage_diagnostics  # type: ignore[assignment]
            result = runtime.registry.get("jarvis_doctor").handler({})
        finally:
            dt._configured_storage_diagnostics = original_configured_storage_diagnostics  # type: ignore[assignment]

    if not result.ok:
        raise SystemExit(f"jarvis doctor should return a conservative readiness packet: {result.output}")
    metadata = result.metadata
    _assert_malformed_diagnostic_flags_fail_closed(metadata)
    if metadata.get("storage_available") is not False:
        raise SystemExit(f"jarvis doctor should not report availability from malformed diagnostics: {metadata}")
    if metadata.get("storage_ready_for_completion_claim") is not False:
        raise SystemExit(f"jarvis doctor should not allow completion claim from malformed diagnostics: {metadata}")
    if metadata.get("storage_recovery_required") is not True:
        raise SystemExit(f"jarvis doctor should require recovery for malformed diagnostics: {metadata}")
    if metadata.get("storage_readiness_blocks_completion_claim") is not True:
        raise SystemExit(f"jarvis doctor should block completion claim for malformed diagnostics: {metadata}")
    if metadata.get("completion_claim_ready") is not False:
        raise SystemExit(f"jarvis doctor should keep completion claim blocked: {metadata}")
    if not any("durable storage recovery required" in blocker for blocker in metadata.get("completion_blockers", [])):
        raise SystemExit(f"jarvis doctor should surface durable storage as a completion blocker: {metadata}")
    proof_queue = metadata.get("completion_proof_queue") or []
    if not proof_queue or proof_queue[0] != "storage status":
        raise SystemExit(f"jarvis doctor should route the next proof through storage status: {metadata}")


def test_doctor_separates_approval_held_recent_runs() -> None:
    class ExplodingRow:
        def __getitem__(self, key: str) -> object:
            raise RuntimeError("secret doctor recent row /\x55sers/example/private/audit.sqlite")

        def __str__(self) -> str:
            raise RuntimeError("secret doctor recent row string /\x55sers/example/private")

    with TemporaryDirectory(prefix="jarvis-doctor-approval-held-") as temp:
        runtime = make_temp_runtime(Path(temp))
        original_recent = runtime.store.recent_tool_runs
        runtime.store.recent_tool_runs = lambda limit=40: [
            ExplodingRow(),
            {
                "id": 21,
                "tool_name": "send_telegram",
                "ok": False,
                "risk": "HIGH_RISK",
                "approved": False,
                "metadata": "{}",
            },
            {
                "id": 22,
                "tool_name": "dispatch_decision_packet",
                "ok": False,
                "risk": "HIGH_RISK",
                "approved": False,
                "approval_id": 22,
                "metadata": '{"failure_kind":"approval-gate"}',
            },
            {
                "id": 23,
                "tool_name": "readiness_report",
                "ok": True,
                "risk": "READ_ONLY",
                "approved": False,
                "metadata": "{}",
            },
        ]
        try:
            result = runtime.registry.get("jarvis_doctor").handler({})
        finally:
            runtime.store.recent_tool_runs = original_recent

    if not result.ok:
        raise SystemExit(f"jarvis doctor approval-held fixture failed: {result.output}")
    metadata = result.metadata
    if metadata.get("recent_tool_runs") != 4:
        raise SystemExit(f"jarvis doctor should preserve total recent run count: {metadata}")
    if metadata.get("readable_recent_tool_runs") != 3:
        raise SystemExit(f"jarvis doctor missed readable recent run count: {metadata}")
    if metadata.get("unreadable_recent_tool_run_rows") != 1:
        raise SystemExit(f"jarvis doctor missed unreadable recent run count: {metadata}")
    expected_audit_review = ["storage status", "recent tool runs", "execution health report"]
    if metadata.get("audit_readability_review_commands") != expected_audit_review:
        raise SystemExit(f"jarvis doctor missed audit readability review commands: {metadata}")
    if metadata.get("audit_readability_review_next_command") != "storage status":
        raise SystemExit(f"jarvis doctor missed audit readability next command: {metadata}")
    if "- unreadable recent tool-run rows: 1" not in result.output:
        raise SystemExit(f"jarvis doctor missed unreadable recent-row output line: {result.output}")
    if "audit readability review: `storage status`, `recent tool runs`, `execution health report`" not in result.output:
        raise SystemExit(f"jarvis doctor missed audit readability output queue: {result.output}")
    if metadata.get("recent_failed_runs") != 1:
        raise SystemExit(f"jarvis doctor should count only true failures as failed: {metadata}")
    if metadata.get("recent_approval_held_runs") != 1:
        raise SystemExit(f"jarvis doctor missed approval-held recent run count: {metadata}")
    if "- recent failed/blocked runs: 1" not in result.output:
        raise SystemExit(f"jarvis doctor missed true-failure output line: {result.output}")
    if "- recent approval-held runs: 1" not in result.output:
        raise SystemExit(f"jarvis doctor missed approval-held output line: {result.output}")
    blockers = metadata.get("completion_blockers") or []
    if "1 recent failed/blocked run(s)" not in blockers:
        raise SystemExit(f"jarvis doctor missed true-failure completion blocker: {metadata}")
    if "1 recent approval-held run(s) need approval review" not in blockers:
        raise SystemExit(f"jarvis doctor missed approval-held completion blocker: {metadata}")
    expected_review = [
        "approval readiness 22",
        "approval packet 22",
        "approval chain proof 22",
        "verification receipt <approved run id from approval chain proof 22>",
    ]
    if metadata.get("approval_held_review_commands") != expected_review:
        raise SystemExit(f"jarvis doctor missed approval-held review commands: {metadata}")
    proof_queue = metadata.get("completion_proof_queue") or []
    for command in expected_audit_review:
        if command not in proof_queue:
            raise SystemExit(f"jarvis doctor completion queue missed audit readability command {command!r}: {metadata}")
    for command in expected_review:
        if command not in proof_queue:
            raise SystemExit(f"jarvis doctor completion queue missed approval-held command {command!r}: {metadata}")
    handoff = metadata.get("doctor_handoff") or {}
    readiness = handoff.get("harness_readiness") or {}
    if readiness.get("recent_failed_runs") != 1 or readiness.get("recent_approval_held_runs") != 1:
        raise SystemExit(f"jarvis doctor handoff missed separated recent-run counts: {metadata}")
    if readiness.get("approval_held_review_commands") != expected_review:
        raise SystemExit(f"jarvis doctor handoff missed approval-held review commands: {metadata}")
    if readiness.get("audit_readability_review_commands") != expected_audit_review:
        raise SystemExit(f"jarvis doctor handoff missed audit readability review commands: {metadata}")
    for leaked in ["/\x55sers/operator", "audit.sqlite", "secret doctor recent row"]:
        if leaked in result.output or leaked in str(metadata):
            raise SystemExit(f"jarvis doctor leaked malformed recent-row text {leaked!r}")


def test_operator_status_readiness_aliases_route_to_read_only_tools() -> None:
    planner = RuleBasedPlanner()
    expected_tools = {
        "status please": "jarvis_status",
        "jarvis status please": "jarvis_status",
        "show latest status": "jarvis_status",
        "diagnostics please": "jarvis_doctor",
        "diagnostic report please": "jarvis_doctor",
        "health please": "jarvis_doctor",
        "health check please": "jarvis_doctor",
        "setup please": "setup_check",
        "setup status please": "setup_check",
        "readiness please": "readiness_report",
        "readiness report please": "readiness_report",
        # Real bug found live 2026-07-09, most severe of the round: "run
        # doctor" / "run diagnostics" fell through every specific handler
        # down to the generic `run <shell command>` catch-all, queueing a
        # HIGH_RISK approval to literally shell-exec a command named "doctor"
        # instead of running Jarvis's own READ_ONLY diagnostic.
        "run doctor": "jarvis_doctor",
        "run diagnostics": "jarvis_doctor",
        "run diagnostic report": "jarvis_doctor",
        "run health check": "jarvis_doctor",
        # Real gap found live 2026-07-09: "what tools ran recently" and
        # "what's my readiness" both fell through to chat while sibling
        # phrasings worked.
        "what tools ran recently": "recent_tool_runs",
        "what's my readiness": "readiness_report",
    }
    for phrase, expected_tool in expected_tools.items():
        plan = planner.plan(phrase)
        action_tools = [action.tool_name for action in plan.actions]
        if action_tools != [expected_tool]:
            raise SystemExit(f"operator alias did not route safely: {phrase!r} -> {action_tools!r}")


def test_runtime_routes_audit_and_status_aliases_without_suggestion_shadow() -> None:
    # Real gap found live 2026-07-09: `assert_planner_routes_*`-style tests
    # above only exercise the bare RuleBasedPlanner, which cannot catch a
    # runtime-level pre-planner-suggestion shadow (same class of bug as the
    # 2026-07-09 "active goals" / "show my goals" find -- see
    # [[jarvis-pre-planner-suggestion-shadow]]). "audit log", "show audit
    # log", and "system status" all resolved correctly via the bare planner
    # but were shadowed in `jarvis_v2/agent/runtime.py`'s
    # `_PRE_PLANNER_SUGGESTION_NORMALIZED` set, turning a one-shot working
    # command into a "Did you mean ...? Send that and I'll run it." loop.
    with TemporaryDirectory(prefix="jarvis-doctor-runtime-alias-") as temp:
        runtime = make_temp_runtime(Path(temp))
        for text, expected_tool in (
            ("audit log", "recent_tool_runs"),
            ("show audit log", "recent_tool_runs"),
            ("system status", "system_info"),
        ):
            result = runtime.handle(text)
            planned = (result.metadata or {}).get("runtime_trace", {}).get("planned_actions") or []
            if [a.get("tool_name") for a in planned] != [expected_tool]:
                raise SystemExit(f"runtime suggestion-shadow regressed for {text!r}: {planned} / {result.response!r}")
        # "audit trail" is also a one-shot read-only audit command now; keep it
        # out of the pre-planner suggestion loop with the audit-log aliases.
        audit_trail_result = runtime.handle("audit trail")
        audit_trail_planned = (audit_trail_result.metadata or {}).get("runtime_trace", {}).get("planned_actions") or []
        if [a.get("tool_name") for a in audit_trail_planned] != ["recent_tool_runs"]:
            raise SystemExit(f"'audit trail' exact-alias route regressed: {audit_trail_planned}")
        if len(runtime.store.list_pending_approvals(limit=100)) != 0:
            raise SystemExit("'audit trail' should stay read-only and queue no approvals")
        # "system health" resolves via a separate, pre-existing runtime-level
        # compact-alias lookup (not the suggestion set, and not touched by
        # this fix) -- confirmed unaffected as a non-regression check.
        system_health_result = runtime.handle("system health")
        system_health_planned = (system_health_result.metadata or {}).get("runtime_trace", {}).get("planned_actions") or []
        if [a.get("tool_name") for a in system_health_planned] != ["jarvis_doctor"]:
            raise SystemExit(f"'system health' compact-alias route regressed: {system_health_planned}")


def main() -> None:
    assert_doctor_metadata_bool_is_exact()
    test_doctor_rejects_malformed_storage_diagnostic_bools()
    test_doctor_separates_approval_held_recent_runs()
    test_operator_status_readiness_aliases_route_to_read_only_tools()
    test_runtime_routes_audit_and_status_aliases_without_suggestion_shadow()
    print("Doctor readiness smoke passed")


if __name__ == "__main__":
    main()
