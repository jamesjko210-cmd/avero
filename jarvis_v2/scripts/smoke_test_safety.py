from __future__ import annotations

import re
from pathlib import Path
from tempfile import TemporaryDirectory

from jarvis_v2.agent.planner import RuleBasedPlanner
from jarvis_v2.automations.compaction import COMPACTION_JOB_NAME, COMPACTION_JOB_TYPE
from jarvis_v2.config import JarvisConfig
from jarvis_v2.scripts.test_runtime import make_temp_runtime
from jarvis_v2.scripts.public_release_candidate import structural_public_candidate_profile
from jarvis_v2.v3_commands import V3_DASHBOARD_COMMAND, V3_DASHBOARD_INFO_COMMAND


IS_PUBLIC_CANDIDATE = structural_public_candidate_profile(Path(__file__).resolve().parents[2]) is not None


def test_planner_routes_safety_status_aliases() -> None:
    # Real gap found live 2026-07-09: "what's my safety status" fell through
    # to chat while bare "safety status" worked.
    p = RuleBasedPlanner()
    for q in ("safety status", "what's my safety status", "what is my safety status"):
        actions = p.plan(q).actions
        if [a.tool_name for a in actions] != ["safety_status"]:
            raise SystemExit(f"safety_status route missed: {q!r} -> {[a.tool_name for a in actions]}")


def test_frozen_routing_risk_report_is_read_only_and_routed() -> None:
    phrases = [
        "redos status",
        "regex dos status",
        "regex risk status",
        "frozen redos finding",
        "frozen regex finding",
        "send routing redos",
        "send call redos",
        "planner redos finding",
        "planner regex risk",
        "what redos is open",
        "what regex dos is open",
        "what is the frozen redos risk",
        "show frozen routing risk",
        "frozen routing risk report",
        "send call routing risk report",
        "redos proof matrix",
        "regex safety status",
        "regex safety report",
        "정규식 위험 상태",
        "레도스 상태",
        "동결 라우팅 위험",
        "전송 라우팅 레도스",
    ]
    with TemporaryDirectory(prefix="jarvis-frozen-routing-risk-") as temp:
        runtime = make_temp_runtime(Path(temp))
        tool = runtime.registry.get("frozen_routing_risk_report")
        if tool.risk.name != "READ_ONLY":
            raise SystemExit(f"frozen routing risk report must stay READ_ONLY: {tool.risk.name}")
        direct = tool.handler({})
        assert_contains(
            direct.output,
            [
                "Jarvis frozen routing risk report",
                "OPEN / deliberately not fixed",
                "KakaoTalk, Telegram, and iMessage",
                "same-shape frozen send/call patterns",
                "frozen send/call routing block",
                "do not edit the frozen send/call routing block",
                "does not inspect source files at runtime",
                "does not",
                "claim the frozen risk fixed",
            ],
            "frozen routing risk report",
        )
        assert_safe_read_only_metadata(direct.metadata, "frozen routing risk report")
        for key in [
            "inspected_source_at_runtime",
            "ran_timing_probe",
            "edited_frozen_code",
            "sends_messages",
            "makes_calls",
        ]:
            if direct.metadata.get(key) is not False:
                raise SystemExit(f"frozen routing risk report should keep {key}=False: {direct.metadata}")
        if direct.metadata.get("finding_status") != "open" or direct.metadata.get("freeze_active") is not True:
            raise SystemExit(f"frozen routing risk report missed open/freeze metadata: {direct.metadata}")

        for phrase in phrases:
            approvals_before = len(runtime.store.list_pending_approvals(limit=100))
            result = runtime.handle(phrase)
            approvals_after = len(runtime.store.list_pending_approvals(limit=100))
            if approvals_after != approvals_before:
                raise SystemExit(f"frozen routing phrase must not queue approvals: {phrase!r}")
            if "Jarvis frozen routing risk report" not in result.response:
                raise SystemExit(f"frozen routing phrase missed report: {phrase!r} -> {result.response!r}")
            tool_results = getattr(result, "tool_results", []) or []
            if len(tool_results) != 1 or getattr(tool_results[0], "tool_name", "") != "frozen_routing_risk_report":
                raise SystemExit(f"frozen routing phrase should run one report tool: {phrase!r} -> {tool_results!r}")
            metadata = getattr(result, "metadata", {}) or {}
            runtime_trace = metadata.get("runtime_trace") or {}
            if runtime_trace.get("approval_required") is True or runtime_trace.get("approval_queue_delta") != 0:
                raise SystemExit(f"frozen routing phrase should stay read-only: {phrase!r} {runtime_trace}")
            planner_metadata = runtime_trace.get("planner_metadata") or {}
            if planner_metadata.get("runtime_exact_tool_alias") is not True:
                raise SystemExit(f"frozen routing phrase should use exact-alias routing: {phrase!r} {planner_metadata}")


def test_planner_input_guard_report_is_read_only_and_routed() -> None:
    phrases = [
        "planner input length guard",
        "input length guard status",
        "long input safety status",
        "long command safety status",
        "planner input cap status",
        "should we add input length guard",
        "total input length guard",
        "planner input guard decision",
        "planner input guard report",
        "long message redos status",
        "long message planner safety",
        "long command redos risk",
        "command length safety report",
        "what is the input length guard decision",
        "what long input risk is open",
        "입력 길이 제한 상태",
        "긴 입력 안전 상태",
        "긴 명령 안전 상태",
        "플래너 입력 제한",
        "긴 메시지 레도스 위험",
    ]
    with TemporaryDirectory(prefix="jarvis-planner-input-guard-") as temp:
        runtime = make_temp_runtime(Path(temp))
        tool = runtime.registry.get("planner_input_guard_report")
        if tool.risk.name != "READ_ONLY":
            raise SystemExit(f"planner input guard report must stay READ_ONLY: {tool.risk.name}")
        direct = tool.handler({})
        assert_contains(
            direct.output,
            [
                "Jarvis planner input guard report",
                "DESIGN DECISION OPEN",
                "no entry-point input-length guard has been implemented",
                "RuleBasedPlanner.plan()",
                "broad behavior change",
                "do not apply a planner-wide cap autonomously",
                "does not inspect source files at runtime",
                "does not",
                "claim the ReDoS fixed",
            ],
            "planner input guard report",
        )
        assert_safe_read_only_metadata(direct.metadata, "planner input guard report")
        for key in [
            "inspected_source_at_runtime",
            "ran_timing_probe",
            "edited_planner",
            "added_input_cap",
            "sends_messages",
            "makes_calls",
        ]:
            if direct.metadata.get(key) is not False:
                raise SystemExit(f"planner input guard report should keep {key}=False: {direct.metadata}")
        if (
            direct.metadata.get("design_status") != "open"
            or direct.metadata.get("input_length_guard_enabled") is not False
            or direct.metadata.get("requires_operator_decision") is not True
        ):
            raise SystemExit(f"planner input guard report missed open/disabled metadata: {direct.metadata}")

        for phrase in phrases:
            approvals_before = len(runtime.store.list_pending_approvals(limit=100))
            result = runtime.handle(phrase)
            approvals_after = len(runtime.store.list_pending_approvals(limit=100))
            if approvals_after != approvals_before:
                raise SystemExit(f"planner input guard phrase must not queue approvals: {phrase!r}")
            if "Jarvis planner input guard report" not in result.response:
                raise SystemExit(f"planner input guard phrase missed report: {phrase!r} -> {result.response!r}")
            tool_results = getattr(result, "tool_results", []) or []
            if len(tool_results) != 1 or getattr(tool_results[0], "tool_name", "") != "planner_input_guard_report":
                raise SystemExit(f"planner input guard phrase should run one report tool: {phrase!r} -> {tool_results!r}")
            metadata = getattr(result, "metadata", {}) or {}
            runtime_trace = metadata.get("runtime_trace") or {}
            if runtime_trace.get("approval_required") is True or runtime_trace.get("approval_queue_delta") != 0:
                raise SystemExit(f"planner input guard phrase should stay read-only: {phrase!r} {runtime_trace}")
            planner_metadata = runtime_trace.get("planner_metadata") or {}
            if planner_metadata.get("runtime_exact_tool_alias") is not True:
                raise SystemExit(f"planner input guard phrase should use exact-alias routing: {phrase!r} {planner_metadata}")


def test_safety_help_surfaces_pending_regex_and_input_guard_reports() -> None:
    with TemporaryDirectory(prefix="jarvis-safety-help-") as temp:
        runtime = make_temp_runtime(Path(temp))
        cases = [
            "help safety",
            "help redos",
            "help regex safety",
            "help frozen routing risk",
            "help planner input guard",
        ]
        for phrase in cases:
            approvals_before = len(runtime.store.list_pending_approvals(limit=100))
            result = runtime.handle(phrase)
            approvals_after = len(runtime.store.list_pending_approvals(limit=100))
            if approvals_after != approvals_before:
                raise SystemExit(f"safety help phrase must not queue approvals: {phrase!r}")
            if "Jarvis help: safety" not in result.response:
                raise SystemExit(f"safety help phrase missed safety topic: {phrase!r} -> {result.response!r}")
            assert_contains(
                result.response,
                [
                    "frozen routing risk report",
                    "planner input guard report",
                    "High-risk actions require explicit approval",
                    "Personal-data and computer-control actions are approval-gated",
                ],
                f"safety help {phrase!r}",
            )
            tool_results = getattr(result, "tool_results", []) or []
            if len(tool_results) != 1 or getattr(tool_results[0], "tool_name", "") != "jarvis_help":
                raise SystemExit(f"safety help phrase should run jarvis_help only: {phrase!r} -> {tool_results!r}")
            runtime_trace = (getattr(result, "metadata", {}) or {}).get("runtime_trace") or {}
            if runtime_trace.get("approval_required") is True or runtime_trace.get("approval_queue_delta") != 0:
                raise SystemExit(f"safety help phrase should stay read-only: {phrase!r} {runtime_trace}")
from jarvis_v2.tools.storage import (
    BOOTSTRAP_CHECK_COMMAND,
    BOOTSTRAP_WRITE_COMMAND,
    STORAGE_RECOVERY_PLAN_COMMAND,
    STORAGE_RECOVERY_CHECK_API,
    STORAGE_RECOVERY_CHECK_COMMAND,
    make_storage_recovery_check_tool,
)


ROOT = Path(__file__).resolve().parents[2]


def assert_contains(text: str, expected: list[str], label: str) -> None:
    missing = [item for item in expected if item not in text]
    if missing:
        raise SystemExit(f"{label} missing expected text: {missing}")


def assert_safe_read_only_metadata(metadata: dict, label: str) -> None:
    for key in [
        "calls_model",
        "executes_tools",
        "calls_external_service",
        "queues_approval",
        "requires_approval",
        "approves_request",
        "dismisses_request",
        "controls_computer",
        "writes_files",
        "writes_database",
        "writes_memory",
        "writes_notes",
        "speaks",
        "reads_clipboard",
        "external_side_effect",
    ]:
        if metadata.get(key) is not False:
            raise SystemExit(f"{label} unsafe metadata {key}: {metadata}")
    if metadata.get("reads_private_data") or metadata.get("reads_personal_data") or metadata.get("executes_side_effect"):
        raise SystemExit(f"{label} should not read private data or execute side effects: {metadata}")


def assert_storage_recovery_check_preserves_configured_issues_under_fallback() -> None:
    with TemporaryDirectory(prefix="jarvis-storage-check-broken-") as temp:
        root = Path(temp)
        missing_parent = root / "missing-parent"
        config = JarvisConfig(
            data_dir=missing_parent,
            db_path=missing_parent / "jarvis.sqlite",
            obsidian_vault=missing_parent / "Vault",
            obsidian_root="Jarvis",
            use_model_planner=False,
        )
        tool = make_storage_recovery_check_tool(
            config,
            lambda: {
                "reason": "primary_storage_not_writable",
                "exception_type": "OperationalError",
                "db_path_display": "workspace-local fallback database",
                "vault_path_display": "workspace-local fallback notes",
            },
        )
        result = tool({})
        metadata = result.metadata
        expected_issues = {
            "database parent does not exist",
            "Obsidian vault does not exist",
            "Obsidian vault is not writable",
            "runtime is using workspace-local fallback storage",
        }
        if result.tool_name != "storage_recovery_check" or not result.ok:
            raise SystemExit(f"broken storage recovery check routed incorrectly: {result}")
        if set(metadata.get("storage_issues") or []) != expected_issues:
            raise SystemExit(f"storage recovery check lost configured issues under fallback: {metadata}")
        if metadata.get("storage_issue_count") != len(expected_issues):
            raise SystemExit(f"storage recovery check issue count diverged: {metadata}")
        blocker = str(metadata.get("storage_recovery_blocker") or "")
        for expected in sorted(expected_issues):
            if expected not in blocker or expected not in result.output:
                raise SystemExit(f"storage recovery check did not preserve blocker detail {expected!r}: {metadata}")
        if metadata.get("storage_recovery_reason") != blocker or metadata.get("storage_readiness_blocker") != blocker:
            raise SystemExit(f"storage recovery check blocker/reason/readiness diverged: {metadata}")
        if metadata.get("storage_recovery_check_passed") is not False:
            raise SystemExit(f"broken storage recovery check should not pass: {metadata}")
        if metadata.get("storage_status") != "needs attention":
            raise SystemExit(f"broken storage recovery check missed storage status alias: {metadata}")
        if metadata.get("storage_available") is not True:
            raise SystemExit(f"broken storage recovery check should expose fallback availability: {metadata}")
        if metadata.get("storage_ready_for_completion_claim") is not False:
            raise SystemExit(f"broken storage recovery check should block completion readiness: {metadata}")
        if metadata.get("storage_active_route") != "workspace-local fallback":
            raise SystemExit(f"broken storage recovery check missed active route alias: {metadata}")
        if metadata.get("storage_runtime_fallback_active") is not True:
            raise SystemExit(f"broken storage recovery check missed fallback active flag: {metadata}")
        assert_safe_read_only_metadata(metadata, "broken storage recovery check")


def assert_readiness_reports_conversation_compaction_state() -> None:
    with TemporaryDirectory(prefix="jarvis-readiness-compaction-") as temp:
        runtime = make_temp_runtime(Path(temp))
        runtime.store.upsert_job("State Snapshot", 360, "state_snapshot", "2099-01-01T00:00:00")

        missing = runtime.registry.get("readiness_report").handler({})
        if not missing.ok:
            raise SystemExit(f"missing-compaction readiness report failed: {missing}")
        if "Conversation Compaction enabled: needs attention | schedule assistant basics" not in missing.output:
            raise SystemExit(f"missing-compaction readiness report missed visible guidance: {missing.output}")
        missing_metadata = missing.metadata
        if missing_metadata.get("conversation_compaction_jobs") != 0:
            raise SystemExit(f"missing-compaction readiness report should show zero jobs: {missing_metadata}")
        if missing_metadata.get("enabled_conversation_compaction_jobs") != 0:
            raise SystemExit(f"missing-compaction readiness report should show zero enabled jobs: {missing_metadata}")
        if missing_metadata.get("conversation_compaction_next_command") != "schedule assistant basics":
            raise SystemExit(f"missing-compaction readiness report pointed to wrong command: {missing_metadata}")

        runtime.store.upsert_job(COMPACTION_JOB_NAME, 1440, COMPACTION_JOB_TYPE, "2099-01-01T00:00:00")
        paused = runtime.registry.get("pause_job").handler({"name": COMPACTION_JOB_NAME})
        if not paused.ok:
            raise SystemExit(f"pause compaction fixture failed: {paused}")
        paused_report = runtime.registry.get("readiness_report").handler({})
        if (
            f"Conversation Compaction enabled: needs attention | resume job {COMPACTION_JOB_NAME}"
            not in paused_report.output
        ):
            raise SystemExit(f"paused-compaction readiness report missed resume guidance: {paused_report.output}")
        paused_metadata = paused_report.metadata
        if paused_metadata.get("conversation_compaction_jobs") != 1:
            raise SystemExit(f"paused-compaction readiness report should show one job: {paused_metadata}")
        if paused_metadata.get("enabled_conversation_compaction_jobs") != 0:
            raise SystemExit(f"paused-compaction readiness report should show zero enabled jobs: {paused_metadata}")
        if paused_metadata.get("disabled_conversation_compaction_jobs") != 1:
            raise SystemExit(f"paused-compaction readiness report should show one disabled job: {paused_metadata}")
        if paused_metadata.get("conversation_compaction_next_command") != f"resume job {COMPACTION_JOB_NAME}":
            raise SystemExit(f"paused-compaction readiness report pointed to wrong command: {paused_metadata}")

        resumed = runtime.registry.get("resume_job").handler({"name": COMPACTION_JOB_NAME})
        if not resumed.ok:
            raise SystemExit(f"resume compaction fixture failed: {resumed}")
        ready = runtime.registry.get("readiness_report").handler({})
        if "Conversation Compaction enabled: ok | list scheduled jobs" not in ready.output:
            raise SystemExit(f"ready-compaction readiness report missed healthy check: {ready.output}")
        ready_metadata = ready.metadata
        if ready_metadata.get("conversation_compaction_jobs") != 1:
            raise SystemExit(f"ready-compaction readiness report should show one job: {ready_metadata}")
        if ready_metadata.get("enabled_conversation_compaction_jobs") != 1:
            raise SystemExit(f"ready-compaction readiness report should show one enabled job: {ready_metadata}")
        if ready_metadata.get("disabled_conversation_compaction_jobs") != 0:
            raise SystemExit(f"ready-compaction readiness report should show zero disabled jobs: {ready_metadata}")
        if ready_metadata.get("conversation_compaction_next_command") != "list scheduled jobs":
            raise SystemExit(f"ready-compaction readiness report pointed to wrong command: {ready_metadata}")


def assert_readiness_report_handles_malformed_local_rows() -> None:
    class ExplodingRow:
        def __getitem__(self, key: str) -> object:
            raise RuntimeError("secret row read /\x55sers/example/private/jarvis.sqlite")

        def __str__(self) -> str:
            raise RuntimeError("secret row string /\x55sers/example/private/message")

    class ExplodingMetadataValue:
        def __bool__(self) -> bool:
            raise RuntimeError("secret metadata bool /\x55sers/example/private/readiness.sqlite")

        def __str__(self) -> str:
            raise RuntimeError("secret metadata string /\x55sers/example/private/readiness.sqlite")

        def __repr__(self) -> str:
            return "<exploding-readiness-metadata>"

    with TemporaryDirectory(prefix="jarvis-readiness-rows-") as temp:
        runtime = make_temp_runtime(Path(temp))
        original_list_jobs = runtime.store.list_jobs
        original_pending = runtime.store.list_pending_approvals
        original_recent = runtime.store.recent_tool_runs
        runtime.store.list_jobs = lambda: [
            ExplodingRow(),
            {
                "name": "/\x55sers/example/private/State Snapshot",
                "job_type": "state_snapshot",
                "enabled": True,
            },
            {
                "name": COMPACTION_JOB_NAME,
                "job_type": COMPACTION_JOB_TYPE,
                "enabled": True,
            },
        ]
        runtime.store.list_pending_approvals = lambda status="pending", limit=25: [
            ExplodingRow(),
            {
                "id": 42,
                "tool_name": "send_telegram /\x55sers/example/private/token",
                "user_input": "message /\x55sers/example/private/contact saying hi",
            },
        ]
        runtime.store.recent_tool_runs = lambda limit=25: [
            ExplodingRow(),
            {
                "id": 9,
                "tool_name": "run_shell_command",
                "ok": False,
                "risk": "HIGH_RISK",
                "approved": False,
                "metadata": "{}",
            },
            {
                "id": 11,
                "tool_name": "dispatch_decision_packet",
                "ok": False,
                "risk": "HIGH_RISK",
                "approved": False,
                "approval_id": 42,
                "metadata": '{"requires_confirmation":"true"}',
            },
            {
                "id": 12,
                "tool_name": "send_kakao",
                "ok": False,
                "risk": "HIGH_RISK",
                "approved": True,
                "metadata": {"failure_stage": ExplodingMetadataValue()},
            },
            {"id": 10, "tool_name": "readiness_report", "ok": True, "risk": "READ_ONLY", "metadata": "{}"},
        ]
        try:
            report = runtime.registry.get("readiness_report").handler({})
        finally:
            runtime.store.list_jobs = original_list_jobs
            runtime.store.list_pending_approvals = original_pending
            runtime.store.recent_tool_runs = original_recent

        if not report.ok:
            raise SystemExit(f"malformed-row readiness report failed: {report}")
        for leaked in [
            "/\x55sers/operator",
            "private",
            "jarvis.sqlite",
            "readiness.sqlite",
            "secret row",
            "secret metadata",
            "message hi",
        ]:
            if leaked in report.output:
                raise SystemExit(f"malformed-row readiness report leaked {leaked}: {report.output}")
        if "Approval details unavailable: one pending row could not be read safely." not in report.output:
            raise SystemExit(f"malformed-row readiness report missed unreadable approval guidance: {report.output}")
        if "Approval #42 send_telegram <local-path>" not in report.output:
            raise SystemExit(f"malformed-row readiness report missed redacted readable approval: {report.output}")
        if "Recent tool-run audit has 1 unreadable row(s); inspect with `recent tool runs`." not in report.output:
            raise SystemExit(f"malformed-row readiness report missed unreadable recent-run guidance: {report.output}")
        if "Scheduled job table has 1 unreadable row(s); inspect with `list scheduled jobs`." not in report.output:
            raise SystemExit(f"malformed-row readiness report missed unreadable scheduled-job guidance: {report.output}")
        if "Recent failed tool runs exist; inspect with `recent tool runs`." not in report.output:
            raise SystemExit(f"malformed-row readiness report missed readable failed-run guidance: {report.output}")
        if (
            "Recent approval-held tool runs exist; review `approval readiness 42` -> `approval packet 42` -> `approval chain proof 42`."
            not in report.output
        ):
            raise SystemExit(f"malformed-row readiness report missed approval-held review chain: {report.output}")
        metadata = report.metadata
        if metadata.get("scheduled_jobs") != 3:
            raise SystemExit(f"malformed-row readiness report should preserve scheduled row count: {metadata}")
        if metadata.get("readable_scheduled_jobs") != 2:
            raise SystemExit(f"malformed-row readiness report missed readable scheduled row count: {metadata}")
        if metadata.get("unreadable_scheduled_job_rows") != 1:
            raise SystemExit(f"malformed-row readiness report missed unreadable scheduled row count: {metadata}")
        if metadata.get("state_snapshot_jobs") != 1 or metadata.get("enabled_state_snapshot_jobs") != 1:
            raise SystemExit(f"malformed-row readiness report lost readable state snapshot job: {metadata}")
        if (
            metadata.get("conversation_compaction_jobs") != 1
            or metadata.get("enabled_conversation_compaction_jobs") != 1
        ):
            raise SystemExit(f"malformed-row readiness report lost readable compaction job: {metadata}")
        if metadata.get("pending_approvals") != 2:
            raise SystemExit(f"malformed-row readiness report should preserve pending row count: {metadata}")
        if metadata.get("recent_tool_runs") != 5:
            raise SystemExit(f"malformed-row readiness report should preserve recent row count: {metadata}")
        if metadata.get("readable_recent_tool_runs") != 4:
            raise SystemExit(f"malformed-row readiness report missed readable recent row count: {metadata}")
        if metadata.get("unreadable_recent_tool_run_rows") != 1:
            raise SystemExit(f"malformed-row readiness report missed unreadable recent row count: {metadata}")
        if metadata.get("recent_failed_runs") != 2:
            raise SystemExit(f"malformed-row readiness report missed readable failed recent row count: {metadata}")
        if metadata.get("recent_approval_held_runs") != 1:
            raise SystemExit(f"malformed-row readiness report missed approval-held recent row count: {metadata}")
        expected_approval_review = [
            "approval readiness 42",
            "approval packet 42",
            "approval chain proof 42",
            "verification receipt <approved run id from approval chain proof 42>",
        ]
        if metadata.get("approval_held_review_commands") != expected_approval_review:
            raise SystemExit(f"malformed-row readiness report missed approval-held review commands: {metadata}")
        if metadata.get("approval_held_review_next_command") != "approval readiness 42":
            raise SystemExit(f"malformed-row readiness report missed approval-held next review command: {metadata}")
        if metadata.get("approval_held_review_approval_id") != "42":
            raise SystemExit(f"malformed-row readiness report missed approval-held approval id: {metadata}")
        assert_safe_read_only_metadata(metadata, "malformed-row readiness report")


def assert_jarvis_doctor_handles_malformed_local_rows() -> None:
    class ExplodingRow:
        def __getitem__(self, key: str) -> object:
            raise RuntimeError("secret doctor row /\x55sers/example/private/doctor.sqlite")

        def __str__(self) -> str:
            raise RuntimeError("secret doctor string /\x55sers/example/private/contact")

    with TemporaryDirectory(prefix="jarvis-doctor-rows-") as temp:
        runtime = make_temp_runtime(Path(temp))
        original_pending = runtime.store.list_pending_approvals
        original_tasks = runtime.store.list_tasks
        original_recent = runtime.store.recent_tool_runs
        runtime.store.list_pending_approvals = lambda status="pending", limit=25: [
            ExplodingRow(),
            {"id": 77, "tool_name": "run_shell_command", "user_input": "run private command"},
        ]
        runtime.store.list_tasks = lambda status="open", limit=20: [ExplodingRow(), {"id": 12}]
        runtime.store.recent_tool_runs = lambda limit=25: [
            ExplodingRow(),
            {
                "id": 9,
                "tool_name": "run_shell_command",
                "ok": False,
                "risk": "HIGH_RISK",
                "approved": False,
                "approval_id": 77,
                "output": "blocked by approval",
                "metadata": '{"failure_stage":"explicit approval required"}',
            },
        ]
        try:
            doctor = runtime.registry.get("jarvis_doctor").handler({})
        finally:
            runtime.store.list_pending_approvals = original_pending
            runtime.store.list_tasks = original_tasks
            runtime.store.recent_tool_runs = original_recent

        if not doctor.ok:
            raise SystemExit(f"malformed-row jarvis doctor failed: {doctor}")
        for leaked in [
            "/\x55sers/operator",
            "private/doctor.sqlite",
            "secret doctor",
            "private/contact",
        ]:
            if leaked in doctor.output:
                raise SystemExit(f"malformed-row jarvis doctor leaked {leaked}: {doctor.output}")
        metadata = doctor.metadata
        if metadata.get("unreadable_recent_tool_run_rows") != 1:
            raise SystemExit(f"malformed-row jarvis doctor missed unreadable recent row count: {metadata}")
        if "1 unreadable recent tool run row(s)" not in (metadata.get("completion_blockers") or []):
            raise SystemExit(f"malformed-row jarvis doctor missed unreadable-row completion blocker: {metadata}")
        proof_queue = metadata.get("completion_proof_queue") or []
        for expected in ["approval readiness 77", "approval packet 77", "approval chain proof 77"]:
            if expected not in proof_queue:
                raise SystemExit(f"malformed-row jarvis doctor missed readable approval proof command {expected}: {metadata}")
        if "task completion packet 12" not in proof_queue:
            raise SystemExit(f"malformed-row jarvis doctor missed readable task proof command: {metadata}")
        if metadata.get("recent_tool_runs") != 2 or metadata.get("readable_recent_tool_runs") != 1:
            raise SystemExit(f"malformed-row jarvis doctor recent row counts diverged: {metadata}")
        if metadata.get("recent_failed_runs") != 0 or metadata.get("recent_approval_held_runs") != 1:
            raise SystemExit(f"malformed-row jarvis doctor missed readable approval-held run: {metadata}")
        expected_held_review = [
            "approval readiness 77",
            "approval packet 77",
            "approval chain proof 77",
            "verification receipt <approved run id from approval chain proof 77>",
        ]
        if metadata.get("approval_held_review_commands") != expected_held_review:
            raise SystemExit(f"malformed-row jarvis doctor missed approval-held review commands: {metadata}")
        if metadata.get("approval_held_review_next_command") != "approval readiness 77":
            raise SystemExit(f"malformed-row jarvis doctor missed approval-held next review command: {metadata}")
        assert_safe_read_only_metadata(metadata, "malformed-row jarvis doctor")
        assert_doctor_handoff(metadata, "malformed-row jarvis doctor")


def assert_safety_status_handles_malformed_pending_approvals() -> None:
    class ExplodingRow:
        def __getitem__(self, key: str) -> object:
            raise RuntimeError("secret safety row /\x55sers/example/private/approval.sqlite")

        def __str__(self) -> str:
            raise RuntimeError("secret safety string /\x55sers/example/private/contact")

    with TemporaryDirectory(prefix="jarvis-safety-status-rows-") as temp:
        runtime = make_temp_runtime(Path(temp))
        original_pending = runtime.store.list_pending_approvals
        runtime.store.list_pending_approvals = lambda status="pending", limit=25: [
            ExplodingRow(),
            {
                "id": 42,
                "tool_name": "send_telegram /\x55sers/example/private/token",
                "user_input": "message /\x55sers/example/private/contact saying hello",
            },
        ]
        try:
            status = runtime.registry.get("safety_status").handler({})
        finally:
            runtime.store.list_pending_approvals = original_pending

        if not status.ok:
            raise SystemExit(f"malformed-row safety status failed: {status}")
        for leaked in [
            "/\x55sers/operator",
            "private/approval.sqlite",
            "secret safety",
            "private/contact",
        ]:
            if leaked in status.output:
                raise SystemExit(f"malformed-row safety status leaked {leaked}: {status.output}")
        if "pending approval row(s) hidden for safety" not in status.output:
            raise SystemExit(f"malformed-row safety status missed hidden-row diagnostic: {status.output}")
        if "#42 send_telegram <local-path>" not in status.output:
            raise SystemExit(f"malformed-row safety status missed redacted readable approval: {status.output}")
        metadata = status.metadata
        if metadata.get("pending_approvals") != 2:
            raise SystemExit(f"malformed-row safety status should preserve total pending count: {metadata}")
        if metadata.get("readable_pending_approvals") != 1:
            raise SystemExit(f"malformed-row safety status missed readable pending count: {metadata}")
        if metadata.get("unreadable_pending_approval_rows") != 1:
            raise SystemExit(f"malformed-row safety status missed unreadable pending count: {metadata}")
        assert_safe_read_only_metadata(metadata, "malformed-row safety status")


def assert_doctor_handoff(metadata: dict, label: str) -> None:
    handoff = metadata.get("doctor_handoff")
    if not isinstance(handoff, dict) or handoff.get("kind") != "doctor_handoff":
        raise SystemExit(f"{label} missed structured doctor handoff: {metadata}")
    if metadata.get("jarvis_doctor_handoff") != handoff:
        raise SystemExit(f"{label} missed canonical jarvis_doctor_handoff alias: {metadata}")
    if metadata.get("jarvis_doctor_handoff_ready") is not True:
        raise SystemExit(f"{label} missed canonical jarvis_doctor_handoff_ready flag: {metadata}")
    if handoff.get("diagnostic_only") is not True or handoff.get("content_in_handoff") is not False:
        raise SystemExit(f"{label} doctor handoff should be diagnostic/content-free: {handoff}")
    boundaries = handoff.get("boundaries")
    if not isinstance(boundaries, dict):
        raise SystemExit(f"{label} doctor handoff missed boundaries: {handoff}")
    for key in [
        "calls_model",
        "calls_external_service",
        "executes_tools",
        "queues_approval",
        "requires_approval",
        "approves_request",
        "dismisses_request",
        "reads_private_data",
        "reads_personal_data",
        "reads_clipboard",
        "executes_side_effect",
        "external_side_effect",
        "controls_computer",
        "writes_files",
        "writes_database",
        "writes_memory",
        "writes_notes",
        "speaks",
        "completes_tasks",
    ]:
        if boundaries.get(key) is not False:
            raise SystemExit(f"{label} doctor handoff unsafe boundary {key}: {handoff}")
        if metadata.get(key) is not False:
            raise SystemExit(f"{label} doctor flat metadata unsafe boundary {key}: {metadata}")
    readiness = handoff.get("harness_readiness") or {}
    if readiness.get("pending_approvals") != metadata.get("pending_approvals"):
        raise SystemExit(f"{label} doctor handoff pending approval parity failed: {handoff}")
    if readiness.get("recent_failed_runs") != metadata.get("recent_failed_runs"):
        raise SystemExit(f"{label} doctor handoff failed-run parity failed: {handoff}")
    if readiness.get("recent_approval_held_runs") != metadata.get("recent_approval_held_runs", 0):
        raise SystemExit(f"{label} doctor handoff approval-held-run parity failed: {handoff}")
    if readiness.get("next_audit_command") != metadata.get("next_audit_command"):
        raise SystemExit(f"{label} doctor handoff next-audit parity failed: {handoff}")
    completion = handoff.get("completion") or {}
    if completion.get("claim_state") != metadata.get("completion_claim_state"):
        raise SystemExit(f"{label} doctor handoff completion state parity failed: {handoff}")
    if completion.get("claim_ready") != metadata.get("completion_claim_ready"):
        raise SystemExit(f"{label} doctor handoff completion readiness parity failed: {handoff}")
    if completion.get("blockers") != metadata.get("completion_blockers"):
        raise SystemExit(f"{label} doctor handoff blockers parity failed: {handoff}")
    if completion.get("proof_queue") != metadata.get("completion_proof_queue"):
        raise SystemExit(f"{label} doctor handoff proof queue parity failed: {handoff}")
    if completion.get("next_proof_command") != metadata.get("next_completion_proof_command"):
        raise SystemExit(f"{label} doctor handoff next proof parity failed: {handoff}")
    recovery = handoff.get("recovery_closure") or {}
    if recovery.get("proof_queue") != metadata.get("execution_health_recovery_closure_proof_queue"):
        raise SystemExit(f"{label} doctor handoff recovery queue parity failed: {handoff}")
    if recovery.get("next_required_command") != metadata.get("execution_health_recovery_closure_next_required_command"):
        raise SystemExit(f"{label} doctor handoff recovery next-required parity failed: {handoff}")
    if recovery.get("next_proof_command") != metadata.get("execution_health_recovery_closure_next_proof_command"):
        raise SystemExit(f"{label} doctor handoff recovery next-proof parity failed: {handoff}")
    if recovery.get("blocks_completion_claim") != metadata.get("execution_health_recovery_closure_blocks_completion_claim"):
        raise SystemExit(f"{label} doctor handoff recovery blocker parity failed: {handoff}")
    learning = handoff.get("execution_learning") or {}
    if learning.get("proof_queue") != metadata.get("execution_learning_proof_queue"):
        raise SystemExit(f"{label} doctor handoff learning queue parity failed: {handoff}")
    if learning.get("next_required_command") != metadata.get("execution_learning_next_required_command"):
        raise SystemExit(f"{label} doctor handoff learning next-required parity failed: {handoff}")
    if learning.get("next_proof_command") != metadata.get("execution_learning_next_proof_command"):
        raise SystemExit(f"{label} doctor handoff learning next-proof parity failed: {handoff}")
    if learning.get("blocks_completion_claim") != metadata.get("execution_learning_blocks_completion_claim"):
        raise SystemExit(f"{label} doctor handoff learning blocker parity failed: {handoff}")
    agi_next = handoff.get("agi_next") or {}
    if agi_next.get("gate") != metadata.get("agi_next_gate"):
        raise SystemExit(f"{label} doctor handoff AGI gate parity failed: {handoff}")
    if agi_next.get("next_build_command") != metadata.get("agi_next_build_command"):
        raise SystemExit(f"{label} doctor handoff AGI command parity failed: {handoff}")
    if agi_next.get("focused_verification_commands") != metadata.get("agi_focused_verification_commands"):
        raise SystemExit(f"{label} doctor handoff AGI verification parity failed: {handoff}")
    for nested_key, flat_key in {
        "review_only": "agi_next_review_only",
        "draft_only": "agi_next_draft_only",
        "loads_without_execution": "agi_next_loads_without_execution",
    }.items():
        if agi_next.get(nested_key) is not True or metadata.get(flat_key) is not True:
            raise SystemExit(f"{label} doctor handoff AGI {nested_key} flag failed: {handoff}")
    for nested_key, flat_key in {
        "authorizes_execution": "agi_next_authorizes_execution",
        "authorizes_completion_claim": "agi_next_authorizes_completion_claim",
        "approval_granted": "agi_next_approval_granted",
    }.items():
        if agi_next.get(nested_key) is not False or metadata.get(flat_key) is not False:
            raise SystemExit(f"{label} doctor handoff AGI no-authority {nested_key} failed: {handoff}")
    for key in [
        "calls_model",
        "executes_tools",
        "writes_files",
        "reads_personal_data",
        "external_side_effect",
        "controls_computer",
        "queues_approval",
    ]:
        if agi_next.get(key) is not False:
            raise SystemExit(f"{label} doctor AGI handoff should keep {key}=False: {handoff}")
    storage = handoff.get("storage") or {}
    if storage.get("available") != metadata.get("storage_available"):
        raise SystemExit(f"{label} doctor handoff storage parity failed: {handoff}")
    if storage.get("sqlite_store_available") != metadata.get("sqlite_store_available"):
        raise SystemExit(f"{label} doctor handoff sqlite parity failed: {handoff}")


def test_env_example_is_safe_and_complete() -> None:
    example = ROOT / ".env.example"
    if not example.exists():
        raise SystemExit(".env.example is missing")
    text = example.read_text()
    required_keys = [
        "JARVIS_V3_ENV",
        "JARVIS_DATA_DIR",
        "JARVIS_DB_PATH",
        "JARVIS_STORAGE_FALLBACK_DIR",
        "JARVIS_DISABLE_STORAGE_FALLBACK",
        "JARVIS_OBSIDIAN_VAULT",
        "OBSIDIAN_VAULT_PATH",
        "JARVIS_CHAT_MODEL",
        "OLLAMA_MODEL",
        "JARVIS_PLANNER_MODEL",
        "JARVIS_MODEL_TIMEOUT_SECONDS",
        "JARVIS_CHAT_TIMEOUT_SECONDS",
        "JARVIS_USE_MODEL_PLANNER",
        "JARVIS_SMOKE_MODULE_TIMEOUT_SECONDS",
        "JARVIS_V3_ENABLE_DAEMONS",
        "JARVIS_V3_ENABLE_SCHEDULER",
        "TELEGRAM_BOT_TOKEN",
        "JARVIS_OWNER_TELEGRAM",
        "GMAIL_ADDRESS",
        "GMAIL_APP_PASSWORD",
        "JARVIS_OWNER_IMESSAGE",
        "JARVIS_GOOGLE_CREDS",
        "JARVIS_GOOGLE_READONLY_TOKEN",
        "JARVIS_GOOGLE_TOKEN",
        "JARVIS_WEATHER_LOCATION",
        "JARVIS_NEWS_LOCALE",
        "JARVIS_REMINDERS_FILE",
        "JARVIS_WATCHED_DIRS",
    ]
    for key in required_keys:
        if f"{key}=" not in text:
            raise SystemExit(f".env.example missed setup key: {key}")
    for key in ["JARVIS_V3_ENV", "TELEGRAM_BOT_TOKEN", "JARVIS_OWNER_TELEGRAM", "GMAIL_ADDRESS", "GMAIL_APP_PASSWORD", "JARVIS_OWNER_IMESSAGE", "JARVIS_OBSIDIAN_VAULT", "OBSIDIAN_VAULT_PATH"]:
        for raw in text.splitlines():
            if raw.startswith(f"{key}=") and raw.partition("=")[2].strip():
                raise SystemExit(f".env.example must keep {key} blank")
    forbidden_fragments = ["bot_token_here", "xox", "AIza", "sk-", "-----BEGIN", "+1555"]
    for fragment in forbidden_fragments:
        if fragment in text:
            raise SystemExit(f".env.example contains secret-looking fragment: {fragment}")
    gitignore = (ROOT / ".gitignore").read_text()
    for expected in [
        ".env",
        ".env.local",
        ".env.*.local",
        "!.env.example",
        "google_credentials.json",
        "google_token.json",
        "google_calendar_readonly_token.json",
        "**/google_credentials.json",
        "**/google_token.json",
        "**/google_calendar_readonly_token.json",
        ".jarvis_v3_runtime/",
        ".jarvis_v3_durable/",
    ]:
        if expected not in gitignore:
            raise SystemExit(f".gitignore missed secret path: {expected}")
    tasks_path = ROOT / "CODEX_TASKS.md"
    if IS_PUBLIC_CANDIDATE:
        if tasks_path.exists():
            raise SystemExit("public candidate retained private operational reference: CODEX_TASKS.md")
    else:
        tasks = tasks_path.read_text()
        for expected in [
            "No secrets or local state in git",
            ".env.local",
            ".env.*.local",
            "nested\n   `google_credentials.json`/`google_token.json`",
            ".jarvis_v2_runtime/",
            ".env.example` stays placeholder-only and tracked",
            "do not create git commits",
        ]:
            if expected not in tasks:
                raise SystemExit(f"CODEX_TASKS missed gitignore setup boundary: {expected}")
        if re.search(r"Do not commit unless [^\n]+ explicitly asks", tasks) is None:
            raise SystemExit("CODEX_TASKS missed the explicit git commit authorization boundary")
    readme = (ROOT / "README.md").read_text()
    readme_normalized = " ".join(readme.split())
    if ".env.example" not in readme or "never put populated environment files in the" not in readme:
        raise SystemExit("README missed safe .env.example setup guidance")
    for expected in [
        "`QUICKSTART.md` is the authoritative fresh-start and daily-use path",
        "Keep local env variants, Google credential/token files, key files,",
        "legacy `.jarvis_v2_runtime/` out of git",
        "No activation command is provided in",
        "`JARVIS_CACHE_DIR` | `~/.cache/jarvis-v3`",
        "sanitized public candidate deliberately omits every service template",
        "Service installation is unavailable in",
        "contains the offline contract generator and content-free recovery receipt helpers",
        "no LaunchAgent templates, installed contracts, installation or restart commands",
        "no `launchctl` or daemon-restart command",
        "JARVIS_SMOKE_MODULE_TIMEOUT_SECONDS",
        "defaulting to `180`",
        "aggregate smoke suite's per-module timeout",
    ]:
        if expected not in readme_normalized:
            raise SystemExit(f"README missed local-state git guidance: {expected}")
    for forbidden in [
        "python3 -m jarvis_v2.scripts.run_scheduler",
        "python3 -m jarvis_v2.scripts.run_subagent_runner",
        "export JARVIS_STATUS_AUTH_TOKEN=",
        "python3 -c 'import secrets",
        "launchctl bootstrap",
        "launchctl kickstart",
        "install -m 600 com.jarvis",
    ]:
        if forbidden in readme_normalized:
            raise SystemExit(f"README exposed a competing preview/service recipe: {forbidden}")
    for expected in [
        "env-presence",
        "Google connector package",
        "credential-file-presence",
        "storage-fallback readiness",
        "without reading clipboard contents, secret values, credential/token file contents, or calling Google",
    ]:
        if expected not in readme:
            raise SystemExit(f"README missed setup diagnostics boundary text: {expected}")


def main() -> None:
    test_planner_routes_safety_status_aliases()
    test_frozen_routing_risk_report_is_read_only_and_routed()
    test_planner_input_guard_report_is_read_only_and_routed()
    test_safety_help_surfaces_pending_regex_and_input_guard_reports()
    test_env_example_is_safe_and_complete()
    assert_storage_recovery_check_preserves_configured_issues_under_fallback()
    assert_readiness_reports_conversation_compaction_state()
    assert_readiness_report_handles_malformed_local_rows()
    assert_jarvis_doctor_handles_malformed_local_rows()
    assert_safety_status_handles_malformed_pending_approvals()
    with TemporaryDirectory(prefix="jarvis-safety-") as temp:
        runtime = make_temp_runtime(Path(temp))
        cases = [
            "help safety",
            "organize brain dump: task: verify Jarvis safety; remember: Jarvis should never harm the operator or his computer",
            "run command python3 --version",
            "write file /tmp/jarvis-safety-test.txt with unsafe direct write attempt",
            "get clipboard",
            "enable computer control",
            "action readiness: run a script and email me the result",
            "safety status",
            "pending approvals",
            "readiness report",
            "storage recovery check",
            "setup check",
            "jarvis doctor",
        ]
        for case in cases:
            result = runtime.handle(case)
            status = "ok" if result.verified else "blocked"
            print(f"[{status}] {case}")
            print(result.response[:1800])
            print()

            if case == "help safety":
                assert_contains(
                    result.response,
                    [
                        "auto-runs only read-only and local-safe tools",
                        "frozen routing risk report",
                        "planner input guard report",
                        "readiness report",
                        "High-risk actions require explicit approval",
                        "Personal-data and computer-control actions are approval-gated",
                    ],
                    "safety help",
                )
            if case.startswith("organize brain dump"):
                assert_contains(result.response, ["tasks: #1", "memories: #"], "safe organizer")
            if case in {"run command python3 --version", "write file /tmp/jarvis-safety-test.txt with unsafe direct write attempt", "get clipboard", "enable computer control"}:
                if result.verified:
                    raise SystemExit(f"Expected '{case}' to be blocked without approval.")
                assert_contains(
                    result.response,
                    ["explicit approval required", "Safety receipt", "pending approvals", "approval readiness", "approval packet", "approval chain proof", "approve approval", "dismiss approval"],
                    case,
                )
                if case == "run command python3 --version":
                    duplicate = runtime.handle(case)
                    if duplicate.tool_results[0].metadata.get("approval_id") != result.tool_results[0].metadata.get("approval_id"):
                        raise SystemExit("Duplicate risky request did not reuse the existing pending approval.")
                    if duplicate.tool_results[0].metadata.get("reused_pending_approval") is not True:
                        raise SystemExit("Duplicate risky request did not mark reused_pending_approval.")
                    assert_contains(
                        duplicate.response,
                        ["Safety receipt: already queued as approval #"],
                        "duplicate approval reuse receipt",
                    )
            if case.startswith("action readiness"):
                assert_contains(
                    result.response,
                    [
                        "Jarvis action readiness packet",
                        "This is read-only",
                        "Recommendation: STOP_AND_REVIEW_APPROVALS",
                        "Detected risk areas",
                        "shell/code",
                        "external side effect",
                        "Current blockers",
                        "approval packet",
                        "approval chain proof",
                        "Missing checks before action",
                        "action rehearsal",
                        "integration action preview",
                        "Go/no-go rule",
                    ],
                    "action readiness",
                )
                metadata = result.tool_results[0].metadata
                if metadata.get("recommendation") != "STOP_AND_REVIEW_APPROVALS":
                    raise SystemExit(f"Expected action readiness to stop for approvals: {metadata}")
                if metadata.get("pending_approvals", 0) < 4:
                    raise SystemExit(f"Expected action readiness to see pending approvals: {metadata}")
                for key in [
                    "calls_model",
                    "executes_tools",
                    "queues_approval",
                    "approves_request",
                    "dismisses_request",
                    "reads_private_data",
                    "writes_files",
                    "controls_computer",
                    "calls_external_services",
                ]:
                    if metadata.get(key) is not False:
                        raise SystemExit(f"Action readiness should report {key}=False.")
            if case == "safety status":
                assert_contains(
                    result.response,
                    [
                        "Jarvis safety status",
                        "auto-run ceiling",
                        "Risk-gated tools",
                        "run_shell_command",
                        "get_clipboard",
                        "enable_computer_control",
                        "approval packet",
                    ],
                    "safety status",
                )
            if case == "pending approvals":
                assert_contains(
                    result.response,
                    ["run_shell_command", "write_text_file", "get_clipboard", "enable_computer_control"],
                    "pending approvals",
                )
                latest_readiness = runtime.handle("approval readiness latest")
                assert_contains(
                    latest_readiness.response,
                    ["Approval readiness packet #", "enable_computer_control", "LAST_LOOK_REQUIRED"],
                    "latest approval readiness",
                )
                if latest_readiness.tool_results[0].metadata.get("approval_id") != 4:
                    raise SystemExit(
                        f"Latest approval readiness resolved the wrong approval: {latest_readiness.tool_results[0].metadata}"
                    )
                latest_packet = runtime.handle("approval packet latest")
                assert_contains(
                    latest_packet.response,
                    ["Approval execution packet #", "enable_computer_control", "Run if trusted"],
                    "latest approval packet",
                )
                if latest_packet.tool_results[0].metadata.get("approval_id") != 4:
                    raise SystemExit(f"Latest approval packet resolved the wrong approval: {latest_packet.tool_results[0].metadata}")
                natural_packet = runtime.handle("preview latest approval")
                if natural_packet.tool_results[0].metadata.get("approval_id") != 4:
                    raise SystemExit(f"Natural approval preview resolved the wrong approval: {natural_packet.tool_results[0].metadata}")
                latest_detail = runtime.handle("approval detail latest")
                assert_contains(
                    latest_detail.response,
                    ["Approval detail #", "enable_computer_control", "Last-look checklist"],
                    "latest approval detail",
                )
                natural_detail = runtime.handle("show latest approval")
                if natural_detail.tool_results[0].metadata.get("approval_id") != 4:
                    raise SystemExit(f"Natural latest approval detail resolved the wrong approval: {natural_detail.tool_results[0].metadata}")
                latest_dismiss = runtime.handle("dismiss approval latest")
                assert_contains(latest_dismiss.response, ["Dismissed approval #4"], "latest approval dismiss")
                latest_queue = runtime.handle("pending approvals")
                if "enable_computer_control" in latest_queue.response:
                    raise SystemExit("Latest approval dismiss did not remove the newest pending approval.")
            if case == "readiness report":
                assert_contains(
                    result.response,
                    [
                        "Jarvis readiness report",
                        "ready with attention",
                        "safe-use rule",
                        "stop-window rule",
                        "priority goals do not override the operator's explicit stop times",
                        "Risk gates registered",
                        "SQLite storage parent writable",
                        "SQLite database file writable",
                        "Approval #",
                        "approval packet",
                        "approval chain proof",
                        "pending approvals",
                        "jarvis doctor",
                    ],
                    "readiness report",
                )
                metadata = result.tool_results[0].metadata
                if metadata.get("storage_available") is not True or metadata.get("storage_metadata_only") is not True:
                    raise SystemExit(f"readiness report missed storage diagnostics metadata: {metadata}")
                if metadata.get("storage_configured_status") != "ready" or metadata.get("storage_configured_available") is not True:
                    raise SystemExit(f"readiness report missed configured storage readiness: {metadata}")
                if metadata.get("storage_runtime_fallback_active") is not False:
                    raise SystemExit(f"readiness report should not see fallback in temp runtime: {metadata}")
                if metadata.get("storage_ready_for_completion_claim") is not True:
                    raise SystemExit(f"readiness report should allow completion storage readiness when fallback is inactive: {metadata}")
                if not metadata.get("storage_db_parent_writable") or not metadata.get("storage_db_file_writable"):
                    raise SystemExit(f"readiness report should see temp storage as writable: {metadata}")
                if metadata.get("operator_timeboxes_override_priority") is not True or metadata.get("stop_times_override_priority") is not True:
                    raise SystemExit(f"readiness report missed operator-limit metadata: {metadata}")
                for key in [
                    "calls_model",
                    "executes_tools",
                    "queues_approval",
                    "approves_request",
                    "dismisses_request",
                    "controls_computer",
                    "reads_private_data",
                    "writes_files",
                    "writes_notes",
                    "writes_memory",
                    "external_side_effect",
                    "requires_approval",
                    "speaks",
                ]:
                    if metadata.get(key) is not False:
                        raise SystemExit(f"readiness report unsafe metadata {key}: {metadata}")
                original_storage_fallback = runtime.storage_fallback
                runtime.storage_fallback = {
                    "reason": "primary_storage_not_writable",
                    "exception_type": "OperationalError",
                    "db_path_display": "workspace-local fallback database",
                    "vault_path_display": "workspace-local fallback notes",
                }
                try:
                    fallback_readiness = runtime.registry.get("readiness_report").handler({})
                finally:
                    runtime.storage_fallback = original_storage_fallback
                if not fallback_readiness.ok:
                    raise SystemExit(f"fallback readiness report failed: {fallback_readiness}")
                if "Runtime storage fallback: needs attention | active; using workspace-local fallback storage" not in fallback_readiness.output:
                    raise SystemExit(f"fallback readiness report missed visible fallback check: {fallback_readiness.output}")
                if "restart or reload Jarvis with the configured durable storage envs" not in fallback_readiness.output:
                    raise SystemExit(f"fallback readiness report missed storage recovery step: {fallback_readiness.output}")
                if STORAGE_RECOVERY_CHECK_COMMAND not in fallback_readiness.output:
                    raise SystemExit(f"fallback readiness report missed native storage recovery check: {fallback_readiness.output}")
                if "restart or reload Jarvis with the configured durable storage envs" not in fallback_readiness.output:
                    raise SystemExit(f"fallback readiness report missed restart-only operator action: {fallback_readiness.output}")
                if "- next required: `storage status`" not in fallback_readiness.output:
                    raise SystemExit(f"fallback readiness report missed visible storage next required command: {fallback_readiness.output}")
                if "- next proof:" in fallback_readiness.output:
                    raise SystemExit(f"fallback readiness report should not use ambiguous storage next proof prose: {fallback_readiness.output}")
                if "- proof queue: `storage status` -> `storage recovery check` -> `storage status`" not in fallback_readiness.output:
                    raise SystemExit(f"fallback readiness report missed visible storage proof queue: {fallback_readiness.output}")
                fallback_metadata = fallback_readiness.metadata
                if fallback_metadata.get("storage_runtime_fallback_active") is not True:
                    raise SystemExit(f"fallback readiness report missed active fallback: {fallback_metadata}")
                if fallback_metadata.get("storage_status") != "needs attention":
                    raise SystemExit(f"fallback readiness report should mark storage attention: {fallback_metadata}")
                if fallback_metadata.get("storage_configured_status") != "ready" or fallback_metadata.get("storage_configured_available") is not True:
                    raise SystemExit(f"fallback readiness report should preserve configured storage status: {fallback_metadata}")
                if fallback_metadata.get("storage_ready_for_completion_claim") is not False:
                    raise SystemExit(f"fallback readiness report should block completion storage readiness: {fallback_metadata}")
                if "runtime is using workspace-local fallback storage" not in (fallback_metadata.get("storage_issues") or []):
                    raise SystemExit(f"fallback readiness report missed fallback issue: {fallback_metadata}")
                if fallback_metadata.get("storage_runtime_fallback_reason") != "primary_storage_not_writable":
                    raise SystemExit(f"fallback readiness report missed fallback reason: {fallback_metadata}")
                if fallback_metadata.get("storage_runtime_fallback_exception_type") != "OperationalError":
                    raise SystemExit(f"fallback readiness report missed fallback exception type: {fallback_metadata}")
                if fallback_metadata.get("storage_recovery_check_command") != BOOTSTRAP_CHECK_COMMAND:
                    raise SystemExit(f"fallback readiness report missed recovery check command: {fallback_metadata}")
                if fallback_metadata.get("storage_recovery_check_api") != STORAGE_RECOVERY_CHECK_API:
                    raise SystemExit(f"fallback readiness report missed recovery check API: {fallback_metadata}")
                if fallback_metadata.get("storage_recovery_command") != BOOTSTRAP_WRITE_COMMAND:
                    raise SystemExit(f"fallback readiness report missed recovery write command: {fallback_metadata}")
                expected_storage_commands = ["storage status", "storage recovery check", "storage status"]
                if fallback_metadata.get("storage_readiness_next_commands") != expected_storage_commands:
                    raise SystemExit(f"fallback readiness report missed storage next commands: {fallback_metadata}")
                if fallback_metadata.get("storage_readiness_next_command_count") != len(expected_storage_commands):
                    raise SystemExit(f"fallback readiness report missed storage next command count: {fallback_metadata}")
                if fallback_metadata.get("storage_readiness_next_required_command") != "storage status":
                    raise SystemExit(f"fallback readiness report missed storage next required command: {fallback_metadata}")
                if fallback_metadata.get("storage_readiness_next_proof_command") != "storage status":
                    raise SystemExit(f"fallback readiness report missed storage next proof command: {fallback_metadata}")
                if fallback_metadata.get("storage_readiness_proof_queue") != expected_storage_commands:
                    raise SystemExit(f"fallback readiness report missed storage proof queue: {fallback_metadata}")
                if fallback_metadata.get("storage_readiness_proof_queue_count") != len(expected_storage_commands):
                    raise SystemExit(f"fallback readiness report missed storage proof queue count: {fallback_metadata}")
                if fallback_metadata.get("storage_readiness_first_proof_command") != "storage status":
                    raise SystemExit(f"fallback readiness report missed first storage proof command: {fallback_metadata}")
                original_storage_fallback = runtime.storage_fallback
                runtime.storage_fallback = {
                    "reason": "primary_storage_not_writable",
                    "exception_type": "OperationalError",
                    "db_path_display": "workspace-local fallback database",
                    "vault_path_display": "workspace-local fallback notes",
                }
                try:
                    fallback_doctor = runtime.registry.get("jarvis_doctor").handler({})
                finally:
                    runtime.storage_fallback = original_storage_fallback
                if not fallback_doctor.ok:
                    raise SystemExit(f"fallback jarvis doctor failed: {fallback_doctor}")
                if "runtime storage fallback: needs attention | active; using workspace-local fallback storage" not in fallback_doctor.output:
                    raise SystemExit(f"fallback jarvis doctor missed visible runtime fallback check: {fallback_doctor.output}")
                if "- storage readiness: needs attention" not in fallback_doctor.output:
                    raise SystemExit(f"fallback jarvis doctor should mark storage readiness as degraded: {fallback_doctor.output}")
                expected_storage_commands = [
                    "storage status",
                    STORAGE_RECOVERY_CHECK_COMMAND,
                    "storage status",
                ]
                if "- storage readiness blocks completion claim: yes" not in fallback_doctor.output:
                    raise SystemExit(f"fallback jarvis doctor missed completion-blocking storage readiness: {fallback_doctor.output}")
                if "- storage readiness next required: `storage status`" not in fallback_doctor.output:
                    raise SystemExit(f"fallback jarvis doctor missed storage next required: {fallback_doctor.output}")
                if "- storage readiness next proof:" in fallback_doctor.output:
                    raise SystemExit(f"fallback jarvis doctor should not use ambiguous storage next proof prose: {fallback_doctor.output}")
                if "- storage next proof:" in fallback_doctor.output:
                    raise SystemExit(f"fallback jarvis doctor should render storage proof alias, not storage next proof: {fallback_doctor.output}")
                if "- storage readiness queue:" not in fallback_doctor.output:
                    raise SystemExit(f"fallback jarvis doctor missed storage readiness queue: {fallback_doctor.output}")
                for expected_command in expected_storage_commands:
                    if f"`{expected_command}`" not in fallback_doctor.output:
                        raise SystemExit(f"fallback jarvis doctor storage queue missed {expected_command}: {fallback_doctor.output}")
                if f"- native recovery check: `{STORAGE_RECOVERY_CHECK_COMMAND}`" not in fallback_doctor.output:
                    raise SystemExit(f"fallback jarvis doctor missed native storage recovery check: {fallback_doctor.output}")
                if "restart or reload Jarvis with the configured durable storage envs" not in fallback_doctor.output:
                    raise SystemExit(f"fallback jarvis doctor missed restart-only operator action: {fallback_doctor.output}")
                doctor_metadata = fallback_doctor.metadata
                if doctor_metadata.get("storage_status") != "needs attention":
                    raise SystemExit(f"fallback jarvis doctor should mark storage attention: {doctor_metadata}")
                if doctor_metadata.get("storage_runtime_fallback_active") is not True:
                    raise SystemExit(f"fallback jarvis doctor missed active fallback metadata: {doctor_metadata}")
                if doctor_metadata.get("storage_ready_for_completion_claim") is not False:
                    raise SystemExit(f"fallback jarvis doctor should block completion storage readiness: {doctor_metadata}")
                if doctor_metadata.get("storage_recovery_required") is not True:
                    raise SystemExit(f"fallback jarvis doctor missed recovery-required metadata: {doctor_metadata}")
                if doctor_metadata.get("storage_recovery_check_command") != BOOTSTRAP_CHECK_COMMAND:
                    raise SystemExit(f"fallback jarvis doctor missed recovery check command: {doctor_metadata}")
                if doctor_metadata.get("storage_recovery_check_api") != STORAGE_RECOVERY_CHECK_API:
                    raise SystemExit(f"fallback jarvis doctor missed recovery check API: {doctor_metadata}")
                if doctor_metadata.get("storage_recovery_check_tool_command") != STORAGE_RECOVERY_CHECK_COMMAND:
                    raise SystemExit(f"fallback jarvis doctor missed native recovery check command: {doctor_metadata}")
                if doctor_metadata.get("storage_recovery_command") != BOOTSTRAP_WRITE_COMMAND:
                    raise SystemExit(f"fallback jarvis doctor missed recovery write command: {doctor_metadata}")
                if doctor_metadata.get("storage_readiness_blocks_completion_claim") is not True:
                    raise SystemExit(f"fallback jarvis doctor missed storage completion blocker: {doctor_metadata}")
                if doctor_metadata.get("storage_readiness_blocker") != doctor_metadata.get("storage_recovery_reason"):
                    raise SystemExit(f"fallback jarvis doctor storage blocker should mirror recovery reason: {doctor_metadata}")
                if doctor_metadata.get("storage_readiness_next_commands") != expected_storage_commands:
                    raise SystemExit(f"fallback jarvis doctor missed storage readiness queue metadata: {doctor_metadata}")
                if BOOTSTRAP_CHECK_COMMAND in doctor_metadata.get("storage_readiness_next_commands", []) or BOOTSTRAP_WRITE_COMMAND in doctor_metadata.get("storage_readiness_next_commands", []):
                    raise SystemExit(f"fallback jarvis doctor restart-only queue should not require bootstrap commands: {doctor_metadata}")
                if doctor_metadata.get("storage_readiness_next_command_count") != len(expected_storage_commands):
                    raise SystemExit(f"fallback jarvis doctor missed storage readiness queue count: {doctor_metadata}")
                if doctor_metadata.get("storage_readiness_next_required_command") != "storage status":
                    raise SystemExit(f"fallback jarvis doctor missed storage readiness next required metadata: {doctor_metadata}")
                if doctor_metadata.get("storage_readiness_next_proof_command") != "storage status":
                    raise SystemExit(f"fallback jarvis doctor missed storage readiness next proof metadata: {doctor_metadata}")
                if doctor_metadata.get("completion_next_proof_command") != "storage status":
                    raise SystemExit(f"fallback jarvis doctor should make storage status the next proof: {doctor_metadata}")
                proof_queue = doctor_metadata.get("completion_proof_queue") or []
                if proof_queue[: len(expected_storage_commands)] != expected_storage_commands:
                    raise SystemExit(f"fallback jarvis doctor proof queue should preserve full storage recovery ladder: {doctor_metadata}")
                doctor_handoff = doctor_metadata.get("doctor_handoff") or {}
                doctor_storage = doctor_handoff.get("storage") or {}
                if doctor_storage.get("status") != "needs attention" or doctor_storage.get("runtime_fallback_active") is not True:
                    raise SystemExit(f"fallback jarvis doctor handoff missed storage fallback: {doctor_handoff}")
                if doctor_storage.get("recovery_check_command") != BOOTSTRAP_CHECK_COMMAND:
                    raise SystemExit(f"fallback jarvis doctor handoff missed recovery check command: {doctor_handoff}")
                if doctor_storage.get("recovery_check_tool_command") != STORAGE_RECOVERY_CHECK_COMMAND:
                    raise SystemExit(f"fallback jarvis doctor handoff missed native recovery check command: {doctor_handoff}")
                if doctor_storage.get("readiness_blocks_completion_claim") is not True:
                    raise SystemExit(f"fallback jarvis doctor handoff missed storage completion blocker: {doctor_handoff}")
                if doctor_storage.get("readiness_next_commands") != expected_storage_commands:
                    raise SystemExit(f"fallback jarvis doctor handoff missed storage readiness queue: {doctor_handoff}")
                if doctor_storage.get("readiness_next_command_count") != len(expected_storage_commands):
                    raise SystemExit(f"fallback jarvis doctor handoff missed storage readiness queue count: {doctor_handoff}")
                if doctor_storage.get("readiness_next_required_command") != "storage status":
                    raise SystemExit(f"fallback jarvis doctor handoff missed storage next required: {doctor_handoff}")
                if doctor_storage.get("readiness_next_proof_command") != "storage status":
                    raise SystemExit(f"fallback jarvis doctor handoff missed storage next proof: {doctor_handoff}")

                repair_fallback = {
                    "reason": "primary_storage_not_writable",
                    "exception_type": "OperationalError",
                    "db_path_display": "workspace-local fallback database",
                    "vault_path_display": "workspace-local fallback notes",
                    "primary_storage_diagnostics": {
                        "available": False,
                        "status": "needs attention",
                        "data_dir": "<local-path>",
                        "db_parent": "<local-path>",
                        "db_path": "<local-path>",
                        "db_exists": True,
                        "data_dir_exists": True,
                        "db_parent_exists": True,
                        "obsidian_vault": "<local-path>",
                        "obsidian_root": "Jarvis",
                        "obsidian_root_path": "<local-path>",
                        "obsidian_vault_exists": True,
                        "obsidian_root_exists": True,
                        "data_dir_writable": False,
                        "db_parent_writable": False,
                        "db_file_writable": False,
                        "obsidian_vault_writable": False,
                        "obsidian_root_writable": False,
                        "workspace_local_notes": False,
                        "issues": [
                            "database parent is not writable",
                            "database file is not writable",
                            "Obsidian vault is not writable",
                        ],
                        "metadata_only": True,
                    },
                }
                repair_expected_commands = [
                    "storage status",
                    STORAGE_RECOVERY_PLAN_COMMAND,
                    STORAGE_RECOVERY_CHECK_COMMAND,
                    BOOTSTRAP_CHECK_COMMAND,
                    BOOTSTRAP_WRITE_COMMAND,
                ]
                runtime.storage_fallback = repair_fallback
                try:
                    repair_readiness = runtime.registry.get("readiness_report").handler({})
                    repair_doctor = runtime.registry.get("jarvis_doctor").handler({})
                finally:
                    runtime.storage_fallback = original_storage_fallback
                if repair_readiness.metadata.get("storage_recovery_mode") != "repair_configured_storage_then_restart_runtime":
                    raise SystemExit(f"fallback readiness report should preserve primary repair mode: {repair_readiness.metadata}")
                if repair_readiness.metadata.get("storage_readiness_next_commands") != repair_expected_commands:
                    raise SystemExit(f"fallback readiness report repair-mode queue diverged: {repair_readiness.metadata}")
                if repair_readiness.metadata.get("storage_readiness_next_required_command") != repair_expected_commands[0]:
                    raise SystemExit(f"fallback readiness report repair-mode missed next required command: {repair_readiness.metadata}")
                if repair_readiness.metadata.get("storage_recovery_check_api") != STORAGE_RECOVERY_CHECK_API:
                    raise SystemExit(f"fallback readiness report repair-mode missed recovery check API: {repair_readiness.metadata}")
                if repair_doctor.metadata.get("storage_recovery_mode") != "repair_configured_storage_then_restart_runtime":
                    raise SystemExit(f"fallback jarvis doctor should preserve primary repair mode: {repair_doctor.metadata}")
                if repair_doctor.metadata.get("storage_readiness_next_commands") != repair_expected_commands:
                    raise SystemExit(f"fallback jarvis doctor repair-mode queue diverged: {repair_doctor.metadata}")
                if repair_doctor.metadata.get("storage_readiness_next_required_command") != repair_expected_commands[0]:
                    raise SystemExit(f"fallback jarvis doctor repair-mode missed next required command: {repair_doctor.metadata}")
                if repair_doctor.metadata.get("storage_recovery_check_api") != STORAGE_RECOVERY_CHECK_API:
                    raise SystemExit(f"fallback jarvis doctor repair-mode missed recovery check API: {repair_doctor.metadata}")
                if repair_doctor.metadata.get("completion_proof_queue", [])[: len(repair_expected_commands)] != repair_expected_commands:
                    raise SystemExit(f"fallback jarvis doctor proof queue should preserve repair-mode ladder: {repair_doctor.metadata}")
                original_list_sessions = runtime.store.list_sessions

                def failing_list_sessions(limit: int = 50):
                    raise OSError("sqlite denied near /\x55sers/example/private/jarvis.sqlite")

                runtime.store.list_sessions = failing_list_sessions
                try:
                    failed_readiness = runtime.registry.get("readiness_report").handler({})
                finally:
                    runtime.store.list_sessions = original_list_sessions
                if not failed_readiness.ok or "SQLite memory store: needs attention | unavailable" not in failed_readiness.output:
                    raise SystemExit(f"readiness report SQLite failure missed stable guidance: {failed_readiness.output}")
                for expected in [
                    "JARVIS_DATA_DIR",
                    "JARVIS_DB_PATH",
                    BOOTSTRAP_CHECK_COMMAND,
                    "then retry",
                ]:
                    if expected not in failed_readiness.output:
                        raise SystemExit(f"readiness report SQLite failure missed actionable guidance {expected}: {failed_readiness.output}")
                if "/\x55sers/operator" in failed_readiness.output or "sqlite denied" in failed_readiness.output:
                    raise SystemExit(f"readiness report leaked raw SQLite exception text: {failed_readiness.output}")
                failed_metadata = failed_readiness.metadata
                if failed_metadata.get("sqlite_memory_store_available") is not False or failed_metadata.get("sqlite_memory_store_exception_type") != "OSError":
                    raise SystemExit(f"readiness report missed SQLite failure diagnostics: {failed_metadata}")
                if failed_metadata.get("status") != "not ready" or failed_metadata.get("readiness_gate_state") != "BLOCKED_REQUIRED_FIXES":
                    raise SystemExit(f"readiness report should block required SQLite failures: {failed_metadata}")
                for key in [
                    "calls_model",
                    "executes_tools",
                    "queues_approval",
                    "approves_request",
                    "dismisses_request",
                    "controls_computer",
                    "reads_private_data",
                    "writes_files",
                    "writes_notes",
                    "writes_memory",
                    "external_side_effect",
                    "requires_approval",
                    "speaks",
                ]:
                    if failed_metadata.get(key) is not False:
                        raise SystemExit(f"failed readiness report unsafe metadata {key}: {failed_metadata}")
            if case == "storage recovery check":
                assert_contains(
                    result.response,
                    [
                        "Jarvis storage recovery check",
                        "Jarvis-native no-write check",
                        "configured storage ready: yes",
                        "runtime fallback active: no",
                        "recovery check passed: yes",
                        BOOTSTRAP_CHECK_COMMAND,
                        "does not create folders, initialize SQLite",
                    ],
                    "storage recovery check",
                )
                metadata = result.tool_results[0].metadata
                if result.tool_results[0].tool_name != "storage_recovery_check":
                    raise SystemExit(f"storage recovery check routed to wrong tool: {result.tool_results[0]}")
                if metadata.get("storage_recovery_check_passed") is not True:
                    raise SystemExit(f"storage recovery check should pass in temp runtime: {metadata}")
                if metadata.get("storage_recovery_required") is not False:
                    raise SystemExit(f"storage recovery check should not require recovery in temp runtime: {metadata}")
                if metadata.get("storage_recovery_check_tool_command") != STORAGE_RECOVERY_CHECK_COMMAND:
                    raise SystemExit(f"storage recovery check missed native command metadata: {metadata}")
                if metadata.get("storage_recovery_check_shell_command") != BOOTSTRAP_CHECK_COMMAND:
                    raise SystemExit(f"storage recovery check missed shell command metadata: {metadata}")
                if metadata.get("storage_recovery_command") != BOOTSTRAP_WRITE_COMMAND:
                    raise SystemExit(f"storage recovery check missed recovery command metadata: {metadata}")
                if metadata.get("next_commands") != [BOOTSTRAP_WRITE_COMMAND, "storage status"]:
                    raise SystemExit(f"storage recovery check missed ready next commands: {metadata}")
                for key in [
                    "calls_model",
                    "executes_tools",
                    "queues_approval",
                    "requires_approval",
                    "approves_request",
                    "dismisses_request",
                    "controls_computer",
                    "reads_private_data",
                    "reads_personal_data",
                    "writes_files",
                    "writes_database",
                    "writes_memory",
                    "writes_notes",
                    "external_side_effect",
                    "authorizes_execution",
                    "authorizes_completion_claim",
                ]:
                    if metadata.get(key) is not False:
                        raise SystemExit(f"storage recovery check should keep {key}=False: {metadata}")
            if case == "setup check":
                v3_shell_prefix = 'JARVIS_V3_ENV="$HOME/.jarvis_v3/runtime.env"'
                dashboard_command = (
                    f"{v3_shell_prefix} ./launch_jarvis_v3_dashboard.py"
                )
                dashboard_status_command = (
                    f'{v3_shell_prefix} ./launch_jarvis_v3.py "status dashboard"'
                )
                readiness_command = (
                    f'{v3_shell_prefix} ./launch_jarvis_v3.py "prototype readiness"'
                )
                assert_contains(
                    result.response,
                    [
                        "Jarvis V3 setup check",
                        "pbpaste",
                        "not executed; clipboard privacy",
                        "TELEGRAM_BOT_TOKEN",
                        "JARVIS_V3_ENV",
                        "JARVIS_DATA_DIR",
                        "JARVIS_DB_PATH",
                        "JARVIS_OBSIDIAN_VAULT",
                        "OBSIDIAN_VAULT_PATH",
                        "JARVIS_OBSIDIAN_ROOT",
                        "JARVIS_OWNER_TELEGRAM",
                        "JARVIS_TELEGRAM_STATE",
                        "GMAIL_ADDRESS",
                        "GMAIL_APP_PASSWORD",
                        "JARVIS_OWNER_IMESSAGE",
                        "JARVIS_V3_IMESSAGE_STATE",
                        "OLLAMA_MODEL",
                        "JARVIS_CHAT_MODEL",
                        "JARVIS_PLANNER_MODEL",
                        "Fallback Ollama model alias validation",
                        "Chat model alias validation",
                        "Planner model alias validation",
                        "JARVIS_MODEL_TIMEOUT_SECONDS",
                        "value hidden",
                        "JARVIS_CHAT_TIMEOUT_SECONDS",
                        "JARVIS_USE_MODEL_PLANNER",
                        "JARVIS_SMOKE_MODULE_TIMEOUT_SECONDS",
                        "JARVIS_WEATHER_LOCATION",
                        "JARVIS_NEWS_LOCALE",
                        "JARVIS_STATUS_HOST",
                        "JARVIS_STATUS_PORT",
                        "Status dashboard host validation",
                        "Status dashboard port validation",
                        "Custom env file validation",
                        "Jarvis data directory validation",
                        "Jarvis SQLite database path validation",
                        "Obsidian vault path validation",
                        "Obsidian Jarvis root folder validation",
                        "Watched directories validation",
                        "Default weather location validation",
                        "News locale validation",
                        "Telegram bot token presence",
                        "Gmail address presence",
                        "Gmail app password presence",
                        "Telegram owner allowlist validation",
                        "iMessage owner allowlist validation",
                        "Telegram command state file validation",
                        "iMessage command state file validation",
                        "Telegram reminders state file validation",
                        "Planner timeout validation",
                        "Chat timeout validation",
                        "Model planner toggle validation",
                        "JARVIS_WATCHED_DIRS",
                        "JARVIS_REMINDERS_FILE",
                        "Google connector dependency (google-auth)",
                        "Google connector dependency (google-auth-oauthlib)",
                        "Google connector dependency (google-auth-httplib2)",
                        "Google connector dependency (google-api-python-client)",
                        "Google credentials file (JARVIS_GOOGLE_CREDS)",
                        "Google Calendar read-only token file (JARVIS_GOOGLE_READONLY_TOKEN)",
                        "Google Calendar full-access mutation token file (JARVIS_GOOGLE_TOKEN)",
                        "contents not read",
                        "Storage fallback (JARVIS_DISABLE_STORAGE_FALLBACK)",
                        "Storage fallback directory parent",
                        "Storage fallback directory target",
                        "path hidden; not created",
                        "KakaoTalk app",
                        "Google Chrome app",
                        "not launched",
                        "Accessibility permission for GUI messaging",
                        "manual check required",
                        "Safe next Terminal commands",
                        "run from the Jarvis V3 project folder",
                        dashboard_command,
                        dashboard_status_command,
                        "jarvis doctor",
                        "prototype readiness",
                        "voice setup check",
                        "computer control status",
                    ],
                    "setup check",
                )
                metadata = result.tool_results[0].metadata
                assert_safe_read_only_metadata(metadata, "setup check")
                if metadata.get("reads_environment") is not True or metadata.get("reads_secret_values") is not False:
                    raise SystemExit(f"setup check should inspect env presence only: {metadata}")
                if metadata.get("reads_google_credentials") is not False:
                    raise SystemExit(f"setup check must not read Google credential files: {metadata}")
                for key in [
                    "google_credentials_present",
                    "google_credentials_configured",
                    "google_credentials_exists",
                    "google_credentials_is_file",
                    "google_credentials_is_dir",
                    "google_credentials_parent_exists",
                    "google_credentials_valid",
                    "google_credentials_source",
                    "google_token_present",
                    "google_token_configured",
                    "google_token_exists",
                    "google_token_is_file",
                    "google_token_is_dir",
                    "google_token_parent_exists",
                    "google_token_valid",
                    "google_token_source",
                ]:
                    if key not in metadata:
                        raise SystemExit(f"setup check missed Google file presence metadata: {metadata}")
                google_deps = metadata.get("google_connector_dependencies")
                if not isinstance(google_deps, dict):
                    raise SystemExit(f"setup check missed Google dependency metadata: {metadata}")
                for dep in [
                    "google-auth",
                    "google-auth-oauthlib",
                    "google-auth-httplib2",
                    "google-api-python-client",
                ]:
                    if dep not in google_deps:
                        raise SystemExit(f"setup check missed Google dependency {dep}: {metadata}")
                for key in [
                    "storage_fallback_disabled",
                    "storage_fallback_configured",
                    "storage_fallback_exists",
                    "storage_fallback_is_dir",
                    "storage_fallback_is_file",
                    "storage_fallback_parent_exists",
                    "storage_fallback_parent_writable",
                    "storage_fallback_valid",
                    "storage_fallback_source",
                    "scans_storage_fallback_dir",
                ]:
                    if key not in metadata:
                        raise SystemExit(f"setup check missed storage fallback metadata {key}: {metadata}")
                if metadata.get("creates_storage_fallback_dir") is not False or metadata.get("scans_storage_fallback_dir") is not False:
                    raise SystemExit(f"setup check must not create or scan storage fallback dirs: {metadata}")
                env_configured = metadata.get("env_configured")
                if not isinstance(env_configured, dict):
                    raise SystemExit(f"setup check missed env configured metadata: {metadata}")
                for env_key in ["JARVIS_STATUS_HOST", "JARVIS_STATUS_PORT"]:
                    if env_key not in env_configured:
                        raise SystemExit(f"setup check missed status dashboard env metadata {env_key}: {metadata}")
                for env_key in ["JARVIS_OWNER_TELEGRAM", "JARVIS_OWNER_IMESSAGE"]:
                    if env_key not in env_configured:
                        raise SystemExit(f"setup check missed owner allowlist env metadata {env_key}: {metadata}")
                if "JARVIS_V3_ENV" not in env_configured:
                    raise SystemExit(f"setup check missed custom env-file env metadata: {metadata}")
                for env_key in ["JARVIS_DATA_DIR", "JARVIS_DB_PATH", "JARVIS_OBSIDIAN_VAULT", "OBSIDIAN_VAULT_PATH", "JARVIS_OBSIDIAN_ROOT"]:
                    if env_key not in env_configured:
                        raise SystemExit(f"setup check missed storage/vault env metadata {env_key}: {metadata}")
                if "JARVIS_WATCHED_DIRS" not in env_configured:
                    raise SystemExit(f"setup check missed watched dirs env metadata: {metadata}")
                if "JARVIS_WEATHER_LOCATION" not in env_configured:
                    raise SystemExit(f"setup check missed weather location env metadata: {metadata}")
                if "JARVIS_NEWS_LOCALE" not in env_configured:
                    raise SystemExit(f"setup check missed news locale env metadata: {metadata}")
                for env_key in ["OLLAMA_MODEL", "JARVIS_CHAT_MODEL", "JARVIS_PLANNER_MODEL"]:
                    if env_key not in env_configured:
                        raise SystemExit(f"setup check missed model alias env metadata {env_key}: {metadata}")
                if "JARVIS_USE_MODEL_PLANNER" not in env_configured:
                    raise SystemExit(f"setup check missed model planner toggle env metadata: {metadata}")
                for env_key in ["JARVIS_TELEGRAM_STATE", "JARVIS_V3_IMESSAGE_STATE", "JARVIS_REMINDERS_FILE"]:
                    if env_key not in env_configured:
                        raise SystemExit(f"setup check missed state-file env metadata {env_key}: {metadata}")
                for key in [
                    "env_file_configured",
                    "env_file_exists",
                    "env_file_is_file",
                    "env_file_is_dir",
                    "env_file_parent_exists",
                    "env_file_valid",
                    "env_file_source",
                    "reads_env_file_contents",
                ]:
                    if key not in metadata:
                        raise SystemExit(f"setup check missed env-file metadata {key}: {metadata}")
                if metadata.get("reads_env_file_contents") is not False:
                    raise SystemExit(f"setup check must not read custom env-file contents: {metadata}")
                for key in [
                    "data_dir_configured",
                    "data_dir_exists",
                    "data_dir_is_dir",
                    "data_dir_is_file",
                    "data_dir_parent_exists",
                    "data_dir_valid",
                    "data_dir_source",
                    "creates_data_dir",
                    "scans_data_dir",
                    "db_path_configured",
                    "db_path_exists",
                    "db_path_is_file",
                    "db_path_is_dir",
                    "db_path_parent_exists",
                    "db_path_valid",
                    "db_path_source",
                    "reads_db_file_contents",
                    "creates_db_file",
                    "obsidian_vault_configured",
                    "obsidian_vault_env_key",
                    "obsidian_vault_exists",
                    "obsidian_vault_is_dir",
                    "obsidian_vault_is_file",
                    "obsidian_vault_parent_exists",
                    "obsidian_vault_valid",
                    "obsidian_vault_source",
                    "creates_obsidian_vault",
                    "scans_obsidian_vault",
                    "obsidian_root_configured",
                    "obsidian_root_valid",
                    "obsidian_root_source",
                    "obsidian_root_chars",
                    "obsidian_root_truncated",
                    "obsidian_root_fallback",
                ]:
                    if key not in metadata:
                        raise SystemExit(f"setup check missed storage/vault metadata {key}: {metadata}")
                for key in [
                    "creates_data_dir",
                    "scans_data_dir",
                    "reads_db_file_contents",
                    "creates_db_file",
                    "creates_obsidian_vault",
                    "scans_obsidian_vault",
                ]:
                    if metadata.get(key) is not False:
                        raise SystemExit(f"setup check must not create/read/scan storage or vault targets: {metadata}")
                if metadata.get("obsidian_root_valid") is not True or metadata.get("obsidian_root_source") not in {"default", "env"}:
                    raise SystemExit(f"setup check should report a safe default/env Obsidian root in this smoke: {metadata}")
                if metadata.get("obsidian_root_fallback") != "Jarvis":
                    raise SystemExit(f"setup check missed Obsidian root fallback metadata: {metadata}")
                for key in [
                    "watched_dirs_configured",
                    "watched_dirs_count",
                    "watched_dirs_existing_dirs",
                    "watched_dirs_missing",
                    "watched_dirs_not_dirs",
                    "watched_dirs_valid",
                    "watched_dirs_source",
                    "scans_watched_dirs",
                ]:
                    if key not in metadata:
                        raise SystemExit(f"setup check missed watched-dirs metadata {key}: {metadata}")
                if metadata.get("scans_watched_dirs") is not False:
                    raise SystemExit(f"setup check must not scan watched directory contents: {metadata}")
                for key in [
                    "weather_location_configured",
                    "weather_location_valid",
                    "weather_location_source",
                    "weather_location_chars",
                    "weather_location_truncated",
                    "weather_location_fallback",
                ]:
                    if key not in metadata:
                        raise SystemExit(f"setup check missed weather location metadata {key}: {metadata}")
                if metadata.get("weather_location_valid") is not True or metadata.get("weather_location_source") not in {"default", "env"}:
                    raise SystemExit(f"setup check should report valid/default weather location in this smoke: {metadata}")
                if metadata.get("weather_location_fallback") != "Seoul":
                    raise SystemExit(f"setup check missed weather fallback metadata: {metadata}")
                for key in [
                    "news_locale_configured",
                    "news_locale_valid",
                    "news_locale_source",
                    "news_locale_raw_chars",
                    "news_locale_raw_truncated",
                    "news_locale_fallback_hl",
                    "news_locale_fallback_gl",
                ]:
                    if key not in metadata:
                        raise SystemExit(f"setup check missed news locale metadata {key}: {metadata}")
                if metadata.get("news_locale_valid") is not True or metadata.get("news_locale_source") not in {"default", "env"}:
                    raise SystemExit(f"setup check should report valid/default news locale in this smoke: {metadata}")
                if metadata.get("news_locale_fallback_hl") != "en-US" or metadata.get("news_locale_fallback_gl") != "US":
                    raise SystemExit(f"setup check missed news locale fallback metadata: {metadata}")
                for prefix, env_key in [
                    ("telegram_bot_token", "TELEGRAM_BOT_TOKEN"),
                    ("gmail_address", "GMAIL_ADDRESS"),
                    ("gmail_app_password", "GMAIL_APP_PASSWORD"),
                ]:
                    for suffix in [
                        "configured",
                        "valid",
                        "source",
                        "required",
                        "value_inspected",
                        "validation_performed",
                    ]:
                        if f"{prefix}_{suffix}" not in metadata:
                            raise SystemExit(f"setup check missed secret env metadata {prefix}_{suffix}: {metadata}")
                    if metadata.get(f"{prefix}_required") is not True:
                        raise SystemExit(f"setup check should mark {env_key} required: {metadata}")
                    if metadata.get(f"{prefix}_valid") is not None:
                        raise SystemExit(f"setup check must not validate {env_key} values: {metadata}")
                    if metadata.get(f"{prefix}_value_inspected") is not False:
                        raise SystemExit(f"setup check inspected {env_key}: {metadata}")
                    if metadata.get(f"{prefix}_validation_performed") is not False:
                        raise SystemExit(f"setup check claimed validation for {env_key}: {metadata}")
                for key in [
                    "telegram_owner_allowlist_configured",
                    "telegram_owner_allowlist_valid",
                    "telegram_owner_allowlist_source",
                    "telegram_owner_allowlist_required",
                    "telegram_owner_allowlist_raw_chars",
                    "telegram_owner_allowlist_raw_truncated",
                    "imessage_owner_allowlist_configured",
                    "imessage_owner_allowlist_valid",
                    "imessage_owner_allowlist_source",
                    "imessage_owner_allowlist_required",
                    "imessage_owner_allowlist_raw_chars",
                    "imessage_owner_allowlist_raw_truncated",
                ]:
                    if key not in metadata:
                        raise SystemExit(f"setup check missed owner allowlist metadata {key}: {metadata}")
                if metadata.get("telegram_owner_allowlist_required") is not True:
                    raise SystemExit(f"setup check should mark Telegram owner as required: {metadata}")
                if metadata.get("telegram_owner_allowlist_configured"):
                    if metadata.get("telegram_owner_allowlist_valid") is not True or metadata.get("telegram_owner_allowlist_source") != "env":
                        raise SystemExit(f"setup check should accept configured Telegram owner values in this smoke: {metadata}")
                else:
                    if metadata.get("telegram_owner_allowlist_valid") is not False or metadata.get("telegram_owner_allowlist_source") != "missing":
                        raise SystemExit(f"setup check should report missing required Telegram owner in this smoke: {metadata}")
                    if "JARVIS_OWNER_TELEGRAM" not in metadata.get("setup_attention", []):
                        raise SystemExit(f"setup check missed missing owner allowlist attention metadata: {metadata}")
                if metadata.get("imessage_owner_allowlist_configured"):
                    if metadata.get("imessage_owner_allowlist_valid") is not True or metadata.get("imessage_owner_allowlist_source") != "env":
                        raise SystemExit(f"setup check should accept configured iMessage owner values in this smoke: {metadata}")
                else:
                    if metadata.get("imessage_owner_allowlist_valid") is not True or metadata.get("imessage_owner_allowlist_source") != "default":
                        raise SystemExit(f"setup check should report unset optional iMessage owner as disabled/default: {metadata}")
                if metadata.get("imessage_owner_allowlist_required") is not False:
                    raise SystemExit(f"setup check should mark iMessage owner as optional: {metadata}")
                for key in [
                    "telegram_state_file_configured",
                    "telegram_state_file_exists",
                    "telegram_state_file_is_file",
                    "telegram_state_file_is_dir",
                    "telegram_state_file_parent_exists",
                    "telegram_state_file_valid",
                    "telegram_state_file_source",
                    "reads_telegram_state_file_contents",
                    "creates_telegram_state_file",
                    "imessage_state_file_configured",
                    "imessage_state_file_exists",
                    "imessage_state_file_is_file",
                    "imessage_state_file_is_dir",
                    "imessage_state_file_parent_exists",
                    "imessage_state_file_valid",
                    "imessage_state_file_source",
                    "reads_imessage_state_file_contents",
                    "creates_imessage_state_file",
                    "reminders_state_file_configured",
                    "reminders_state_file_exists",
                    "reminders_state_file_is_file",
                    "reminders_state_file_is_dir",
                    "reminders_state_file_parent_exists",
                    "reminders_state_file_valid",
                    "reminders_state_file_source",
                    "reads_reminders_state_file_contents",
                    "creates_reminders_state_file",
                ]:
                    if key not in metadata:
                        raise SystemExit(f"setup check missed state-file metadata {key}: {metadata}")
                for key in [
                    "reads_telegram_state_file_contents",
                    "creates_telegram_state_file",
                    "reads_imessage_state_file_contents",
                    "creates_imessage_state_file",
                    "reads_reminders_state_file_contents",
                    "creates_reminders_state_file",
                ]:
                    if metadata.get(key) is not False:
                        raise SystemExit(f"setup check must not read/create local state files during diagnostics: {metadata}")
                for key in ["status_dashboard_host_configured", "status_dashboard_port_configured"]:
                    if key not in metadata:
                        raise SystemExit(f"setup check missed status dashboard metadata {key}: {metadata}")
                if metadata.get("status_dashboard_host_valid") is not True or metadata.get("status_dashboard_host_source") not in {"default", "env"}:
                    raise SystemExit(f"setup check should report a valid default/env status host in this smoke: {metadata}")
                if metadata.get("status_dashboard_host_default") != "127.0.0.1":
                    raise SystemExit(f"setup check missed status dashboard default host metadata: {metadata}")
                if metadata.get("status_dashboard_port_valid") is not True or metadata.get("status_dashboard_port_source") not in {"default", "env"}:
                    raise SystemExit(f"setup check should report a valid default/env status port in this smoke: {metadata}")
                if metadata.get("status_dashboard_port_default") != 8766:
                    raise SystemExit(f"setup check missed status dashboard default port metadata: {metadata}")
                for key in [
                    "planner_timeout_configured",
                    "planner_timeout_valid",
                    "planner_timeout_source",
                    "planner_timeout_default",
                    "planner_timeout_effective_seconds",
                    "chat_timeout_configured",
                    "chat_timeout_valid",
                    "chat_timeout_source",
                    "chat_timeout_default",
                    "chat_timeout_effective_seconds",
                    "smoke_module_timeout_configured",
                    "smoke_module_timeout_valid",
                    "smoke_module_timeout_source",
                    "smoke_module_timeout_default",
                    "smoke_module_timeout_effective_seconds",
                    "model_planner_toggle_configured",
                    "model_planner_toggle_valid",
                    "model_planner_toggle_source",
                    "model_planner_toggle_effective_enabled",
                    "model_planner_toggle_fallback_enabled",
                    "fallback_model_alias_configured",
                    "fallback_model_alias_valid",
                    "fallback_model_alias_source",
                    "fallback_model_alias_chars",
                    "fallback_model_alias_truncated",
                    "fallback_model_alias_fallback",
                    "chat_model_alias_configured",
                    "chat_model_alias_valid",
                    "chat_model_alias_source",
                    "chat_model_alias_chars",
                    "chat_model_alias_truncated",
                    "chat_model_alias_fallback",
                    "planner_model_alias_configured",
                    "planner_model_alias_valid",
                    "planner_model_alias_source",
                    "planner_model_alias_chars",
                    "planner_model_alias_truncated",
                    "planner_model_alias_fallback",
                ]:
                    if key not in metadata:
                        raise SystemExit(f"setup check missed timeout/model-planner metadata {key}: {metadata}")
                if metadata.get("planner_timeout_valid") is not True or metadata.get("planner_timeout_source") not in {"default", "env", "env-clamped"}:
                    raise SystemExit(f"setup check should report valid/default planner timeout in this smoke: {metadata}")
                if metadata.get("planner_timeout_default") != 2.5:
                    raise SystemExit(f"setup check missed planner timeout default metadata: {metadata}")
                if metadata.get("chat_timeout_valid") is not True or metadata.get("chat_timeout_source") not in {"default", "env", "env-clamped"}:
                    raise SystemExit(f"setup check should report valid/default chat timeout in this smoke: {metadata}")
                if metadata.get("chat_timeout_default") != 20.0:
                    raise SystemExit(f"setup check missed chat timeout default metadata: {metadata}")
                if metadata.get("smoke_module_timeout_valid") is not True or metadata.get("smoke_module_timeout_source") not in {"default", "env", "env-clamped"}:
                    raise SystemExit(f"setup check should report valid/default smoke timeout in this smoke: {metadata}")
                if metadata.get("smoke_module_timeout_default") != 180:
                    raise SystemExit(f"setup check missed smoke timeout default metadata: {metadata}")
                if metadata.get("model_planner_toggle_valid") is not True or metadata.get("model_planner_toggle_source") not in {"default", "env"}:
                    raise SystemExit(f"setup check should report valid/default model planner toggle in this smoke: {metadata}")
                if metadata.get("model_planner_toggle_fallback_enabled") is not True:
                    raise SystemExit(f"setup check missed model planner toggle fallback metadata: {metadata}")
                if metadata.get("fallback_model_alias_valid") is not True or metadata.get("fallback_model_alias_source") not in {"default", "env"}:
                    raise SystemExit(f"setup check should report valid/default fallback model alias in this smoke: {metadata}")
                if metadata.get("fallback_model_alias_fallback") != "llama3.1":
                    raise SystemExit(f"setup check missed fallback model alias default metadata: {metadata}")
                if metadata.get("chat_model_alias_valid") is not True or metadata.get("chat_model_alias_source") not in {"default", "env"}:
                    raise SystemExit(f"setup check should report valid/default chat model alias in this smoke: {metadata}")
                if metadata.get("planner_model_alias_valid") is not True or metadata.get("planner_model_alias_source") not in {"default", "env"}:
                    raise SystemExit(f"setup check should report valid/default planner model alias in this smoke: {metadata}")
                if metadata.get("creates_storage_fallback_dir") is not False:
                    raise SystemExit(f"setup check must not create storage fallback dirs: {metadata}")
                if metadata.get("launches_apps") is not False:
                    raise SystemExit(f"setup check must not launch apps: {metadata}")
                if metadata.get("telegram_owner_allowlist_valid") is False and "JARVIS_OWNER_TELEGRAM" not in metadata.get("setup_attention", []):
                    raise SystemExit(f"setup check missed owner allowlist attention metadata: {metadata}")
                for key in ["TELEGRAM_BOT_TOKEN", "GMAIL_ADDRESS", "GMAIL_APP_PASSWORD"]:
                    if not metadata.get("env_configured", {}).get(key) and key not in metadata.get("setup_attention", []):
                        raise SystemExit(f"setup check missed required secret attention metadata for {key}: {metadata}")
                for command in [
                    dashboard_command,
                    dashboard_status_command,
                    readiness_command,
                ]:
                    if command not in metadata.get("next_commands", []):
                        raise SystemExit(f"setup check missed next command metadata: {metadata}")
            if case == "jarvis doctor":
                assert_contains(
                    result.response,
                    [
                        "Jarvis doctor",
                        "pbpaste",
                        "clipboard was not read",
                        "Google connector dependency (google-auth)",
                        "Google connector dependency (google-auth-oauthlib)",
                        "Google connector dependency (google-auth-httplib2)",
                        "Google connector dependency (google-api-python-client)",
                        "credentials not read",
                        "Next actions",
                        "Safe next commands",
                        "Project discovery",
                        "active project: Jarvis V3",
                        "not `jarvis-ollama`",
                        "dashboard launcher",
                        V3_DASHBOARD_COMMAND,
                        V3_DASHBOARD_INFO_COMMAND,
                        "Telegram bot token env (TELEGRAM_BOT_TOKEN)",
                        "Telegram bot token validation",
                        "Gmail address env (GMAIL_ADDRESS)",
                        "Gmail address validation",
                        "Gmail app password env (GMAIL_APP_PASSWORD)",
                        "Gmail app password validation",
                        "fallback Ollama model alias env (OLLAMA_MODEL)",
                        "fallback Ollama model alias validation",
                        "chat model alias env (JARVIS_CHAT_MODEL)",
                        "chat model alias validation",
                        "planner model alias env (JARVIS_PLANNER_MODEL)",
                        "planner model alias validation",
                        "planner timeout env (JARVIS_MODEL_TIMEOUT_SECONDS)",
                        "planner timeout validation",
                        "chat timeout env (JARVIS_CHAT_TIMEOUT_SECONDS)",
                        "chat timeout validation",
                        "model planner toggle env (JARVIS_USE_MODEL_PLANNER)",
                        "model planner toggle validation",
                        "smoke module timeout env (JARVIS_SMOKE_MODULE_TIMEOUT_SECONDS)",
                        "smoke module timeout validation",
                        "dashboard host env (JARVIS_STATUS_HOST)",
                        "dashboard host validation",
                        "dashboard port env (JARVIS_STATUS_PORT)",
                        "dashboard port validation",
                        "value hidden",
                        "setup check",
                        "prototype readiness",
                        "readiness report",
                        "Harness readiness diagnostics",
                        "Storage diagnostics",
                        "database parent writable",
                        "database file writable",
                        "pending approvals",
                        "recent failed/blocked runs",
                        "recent verification/audit packets",
                        "next audit command",
                        "recovery closure state",
                        "recovery closure next required",
                        "recovery closure proof queue",
                        "execution learning next required",
                        "completion claim state",
                        "completion blockers",
                        "AGI harness completion handoff",
                        "next AGI gate",
                        "selection source: harness_dynamic_registry",
                        "same current registry-evidence AGI selector",
                        "target file check: TARGETS_EXIST",
                        "missing target files: none",
                        "evidence closure commands",
                        "focused verification",
                        "agi next build move:",
                        "completion claim gate",
                        "Boundary",
                        "diagnostic only",
                    ],
                    "jarvis doctor",
                )
                metadata = result.tool_results[0].metadata
                doctor_output = result.tool_results[0].output
                assert_safe_read_only_metadata(metadata, "jarvis doctor")
                assert_doctor_handoff(metadata, "jarvis doctor")
                if "recovery closure next proof:" in doctor_output:
                    raise SystemExit(f"jarvis doctor should not use ambiguous recovery closure next proof prose: {doctor_output}")
                if "execution learning next proof:" in doctor_output:
                    raise SystemExit(f"jarvis doctor should not use ambiguous execution learning next proof prose: {doctor_output}")
                if "storage next proof:" in doctor_output:
                    raise SystemExit(f"jarvis doctor should render storage proof alias, not storage next proof: {doctor_output}")
                if metadata.get("storage_available") is not True or metadata.get("storage_metadata_only") is not True:
                    raise SystemExit(f"jarvis doctor missed storage diagnostics metadata: {metadata}")
                google_deps = metadata.get("google_connector_dependencies")
                if not isinstance(google_deps, dict):
                    raise SystemExit(f"jarvis doctor missed Google dependency metadata: {metadata}")
                for dep in [
                    "google-auth",
                    "google-auth-oauthlib",
                    "google-auth-httplib2",
                    "google-api-python-client",
                ]:
                    if dep not in google_deps:
                        raise SystemExit(f"jarvis doctor missed Google dependency {dep}: {metadata}")
                if metadata.get("reads_google_credentials") is not False:
                    raise SystemExit(f"jarvis doctor must not read Google credential files: {metadata}")
                if metadata.get("project_name") != "Jarvis V3" or metadata.get("project_root") != "<local-path>":
                    raise SystemExit(f"jarvis doctor missed project discovery metadata: {metadata}")
                if metadata.get("dashboard_launcher") != "<local-path>" or metadata.get("path_metadata_redacted") is not True:
                    raise SystemExit(f"jarvis doctor should expose redacted project path metadata: {metadata}")
                if metadata.get("dashboard_launcher_exists") is not True or metadata.get("dashboard_launch_command") != V3_DASHBOARD_COMMAND:
                    raise SystemExit(f"jarvis doctor missed dashboard launcher metadata: {metadata}")
                if metadata.get("dashboard_ask_command") != V3_DASHBOARD_INFO_COMMAND:
                    raise SystemExit(f"jarvis doctor missed dashboard ask command metadata: {metadata}")
                for key in [
                    "planner_timeout_configured",
                    "planner_timeout_valid",
                    "planner_timeout_source",
                    "planner_timeout_default",
                    "planner_timeout_effective_seconds",
                    "chat_timeout_configured",
                    "chat_timeout_valid",
                    "chat_timeout_source",
                    "chat_timeout_default",
                    "chat_timeout_effective_seconds",
                    "smoke_module_timeout_configured",
                    "smoke_module_timeout_valid",
                    "smoke_module_timeout_source",
                    "smoke_module_timeout_default",
                    "smoke_module_timeout_effective_seconds",
                    "model_planner_toggle_configured",
                    "model_planner_toggle_valid",
                    "model_planner_toggle_source",
                    "model_planner_toggle_effective_enabled",
                    "model_planner_toggle_fallback_enabled",
                    "fallback_model_alias_configured",
                    "fallback_model_alias_valid",
                    "fallback_model_alias_source",
                    "fallback_model_alias_chars",
                    "fallback_model_alias_truncated",
                    "fallback_model_alias_fallback",
                    "chat_model_alias_configured",
                    "chat_model_alias_valid",
                    "chat_model_alias_source",
                    "chat_model_alias_chars",
                    "chat_model_alias_truncated",
                    "chat_model_alias_fallback",
                    "planner_model_alias_configured",
                    "planner_model_alias_valid",
                    "planner_model_alias_source",
                    "planner_model_alias_chars",
                    "planner_model_alias_truncated",
                    "planner_model_alias_fallback",
                    "telegram_bot_token_configured",
                    "telegram_bot_token_valid",
                    "telegram_bot_token_source",
                    "telegram_bot_token_required",
                    "telegram_bot_token_raw_chars",
                    "telegram_bot_token_raw_truncated",
                    "gmail_address_configured",
                    "gmail_address_valid",
                    "gmail_address_source",
                    "gmail_address_required",
                    "gmail_address_raw_chars",
                    "gmail_address_raw_truncated",
                    "gmail_app_password_configured",
                    "gmail_app_password_valid",
                    "gmail_app_password_source",
                    "gmail_app_password_required",
                    "gmail_app_password_raw_chars",
                    "gmail_app_password_raw_truncated",
                ]:
                    if key not in metadata:
                        raise SystemExit(f"jarvis doctor missed timeout/model-planner metadata {key}: {metadata}")
                for prefix in ["telegram_bot_token", "gmail_address", "gmail_app_password"]:
                    if metadata.get(f"{prefix}_required") is not True:
                        raise SystemExit(f"jarvis doctor should mark {prefix} required: {metadata}")
                    if metadata.get(f"{prefix}_source") not in {"env", "missing", "env-invalid"}:
                        raise SystemExit(f"jarvis doctor reported unexpected {prefix} source: {metadata}")
                if metadata.get("planner_timeout_valid") is not True or metadata.get("planner_timeout_source") not in {"default", "env", "env-clamped"}:
                    raise SystemExit(f"jarvis doctor should report valid/default planner timeout in this smoke: {metadata}")
                if metadata.get("planner_timeout_default") != 2.5:
                    raise SystemExit(f"jarvis doctor missed planner timeout default metadata: {metadata}")
                if metadata.get("chat_timeout_valid") is not True or metadata.get("chat_timeout_source") not in {"default", "env", "env-clamped"}:
                    raise SystemExit(f"jarvis doctor should report valid/default chat timeout in this smoke: {metadata}")
                if metadata.get("chat_timeout_default") != 20.0:
                    raise SystemExit(f"jarvis doctor missed chat timeout default metadata: {metadata}")
                if metadata.get("smoke_module_timeout_valid") is not True or metadata.get("smoke_module_timeout_source") not in {"default", "env", "env-clamped"}:
                    raise SystemExit(f"jarvis doctor should report valid/default smoke timeout in this smoke: {metadata}")
                if metadata.get("smoke_module_timeout_default") != 180:
                    raise SystemExit(f"jarvis doctor missed smoke timeout default metadata: {metadata}")
                if metadata.get("model_planner_toggle_valid") is not True or metadata.get("model_planner_toggle_source") not in {"default", "env"}:
                    raise SystemExit(f"jarvis doctor should report valid/default model planner toggle in this smoke: {metadata}")
                if metadata.get("model_planner_toggle_fallback_enabled") is not True:
                    raise SystemExit(f"jarvis doctor missed model planner toggle fallback metadata: {metadata}")
                if metadata.get("fallback_model_alias_valid") is not True or metadata.get("fallback_model_alias_source") not in {"default", "env"}:
                    raise SystemExit(f"jarvis doctor should report valid/default fallback model alias in this smoke: {metadata}")
                if metadata.get("fallback_model_alias_fallback") != "llama3.1":
                    raise SystemExit(f"jarvis doctor missed fallback model alias default metadata: {metadata}")
                if metadata.get("chat_model_alias_valid") is not True or metadata.get("chat_model_alias_source") not in {"default", "env"}:
                    raise SystemExit(f"jarvis doctor should report valid/default chat model alias in this smoke: {metadata}")
                if metadata.get("planner_model_alias_valid") is not True or metadata.get("planner_model_alias_source") not in {"default", "env"}:
                    raise SystemExit(f"jarvis doctor should report valid/default planner model alias in this smoke: {metadata}")
                for key in ["status_dashboard_host_configured", "status_dashboard_port_configured", "status_dashboard_port_valid", "status_dashboard_port_source", "status_dashboard_port_default"]:
                    if key not in metadata:
                        raise SystemExit(f"jarvis doctor missed status dashboard env metadata {key}: {metadata}")
                for key in ["status_dashboard_host_valid", "status_dashboard_host_source", "status_dashboard_host_default"]:
                    if key not in metadata:
                        raise SystemExit(f"jarvis doctor missed status dashboard env metadata {key}: {metadata}")
                if metadata.get("status_dashboard_host_valid") is not True or metadata.get("status_dashboard_host_source") not in {"default", "env"}:
                    raise SystemExit(f"jarvis doctor should report a valid default/env status host in this smoke: {metadata}")
                if metadata.get("status_dashboard_host_default") != "127.0.0.1":
                    raise SystemExit(f"jarvis doctor missed status dashboard default host metadata: {metadata}")
                if metadata.get("status_dashboard_port_valid") is not True or metadata.get("status_dashboard_port_source") not in {"default", "env"}:
                    raise SystemExit(f"jarvis doctor should report a valid default/env status port in this smoke: {metadata}")
                if metadata.get("status_dashboard_port_default") != 8766:
                    raise SystemExit(f"jarvis doctor missed status dashboard default port metadata: {metadata}")
                if not metadata.get("storage_db_parent_writable") or not metadata.get("storage_db_file_writable"):
                    raise SystemExit(f"jarvis doctor should see temp storage as writable: {metadata}")
                if "readiness report" not in metadata.get("next_commands", []):
                    raise SystemExit(f"jarvis doctor missed next command metadata: {metadata}")
                if "completion claim gate" not in metadata.get("next_commands", []):
                    raise SystemExit(f"jarvis doctor missed completion gate next command: {metadata}")
                if metadata.get("agi_handoff_selection_source") != "harness_dynamic_registry":
                    raise SystemExit(f"jarvis doctor should use dynamic AGI registry selection: {metadata}")
                if not metadata.get("agi_next_gate"):
                    raise SystemExit(f"jarvis doctor missed AGI next gate metadata: {metadata}")
                if metadata.get("agi_next_gate") not in str(metadata.get("agi_next_build_command") or ""):
                    raise SystemExit(f"jarvis doctor missed AGI next build command: {metadata}")
                if metadata.get("agi_target_file_integrity_status") != "TARGETS_EXIST":
                    raise SystemExit(f"jarvis doctor should prove AGI target files exist: {metadata}")
                if metadata.get("agi_target_files_exist") is not True or metadata.get("agi_missing_target_files") != [] or metadata.get("agi_missing_target_file_count") != 0:
                    raise SystemExit(f"jarvis doctor reported stale AGI target files: {metadata}")
                if metadata.get("agi_target_files_checked") != len(metadata.get("agi_likely_files", [])):
                    raise SystemExit(f"jarvis doctor AGI target file count diverged: {metadata}")
                if any(not row.get("exists") for row in metadata.get("agi_target_file_rows", [])):
                    raise SystemExit(f"jarvis doctor AGI target file rows include missing files: {metadata}")
                closure_commands = metadata.get("agi_evidence_closure_commands") or []
                expected_claim_gate = f"completion claim gate: improve AGI gate {metadata.get('agi_next_gate')}"
                if "evidence ledger" not in closure_commands or expected_claim_gate not in closure_commands:
                    raise SystemExit(f"jarvis doctor missed AGI evidence closure commands: {metadata}")
                if metadata.get("agi_evidence_closure_command_count") != len(closure_commands) or len(closure_commands) < 4:
                    raise SystemExit(f"jarvis doctor AGI closure count is wrong: {metadata}")
                focused_verification = metadata.get("agi_focused_verification_commands") or []
                if not focused_verification:
                    raise SystemExit(f"jarvis doctor missed focused AGI verification: {metadata}")
                if metadata.get("pending_approvals", 0) < 1:
                    raise SystemExit(f"jarvis doctor missed pending approval diagnostics: {metadata}")
                if metadata.get("recent_failed_runs", 0) + metadata.get("recent_approval_held_runs", 0) < 1:
                    raise SystemExit(f"jarvis doctor missed unresolved recent-run diagnostics: {metadata}")
                if not metadata.get("next_audit_command"):
                    raise SystemExit(f"jarvis doctor missed next audit command metadata: {metadata}")
                if not metadata.get("execution_health_recovery_closure_state"):
                    raise SystemExit(f"jarvis doctor missed recovery closure state metadata: {metadata}")
                recovery_commands = metadata.get("execution_health_recovery_closure_required_commands") or []
                if metadata.get("execution_health_recovery_closure_proof_queue") != recovery_commands:
                    raise SystemExit(f"jarvis doctor recovery closure proof queue should mirror required commands: {metadata}")
                if metadata.get("execution_health_recovery_closure_proof_queue_count") != len(recovery_commands):
                    raise SystemExit(f"jarvis doctor recovery closure proof queue count diverged: {metadata}")
                if metadata.get("recent_failed_runs", 0) > 0:
                    if metadata.get("execution_health_recovery_closure_blocks_completion_claim") is not True:
                        raise SystemExit(f"jarvis doctor should report true failures as recovery-closure blockers: {metadata}")
                    if metadata.get("execution_health_recovery_closure_missing_count", 0) < 1:
                        raise SystemExit(f"jarvis doctor missed recovery closure missing proof metadata: {metadata}")
                    if not recovery_commands or metadata.get("execution_health_recovery_closure_next_required_command") != recovery_commands[0]:
                        raise SystemExit(f"jarvis doctor missed recovery closure proof queue metadata: {metadata}")
                    if metadata.get("execution_health_recovery_closure_next_proof_command") != recovery_commands[0]:
                        raise SystemExit(f"jarvis doctor missed next recovery closure proof command: {metadata}")
                    for command in ["verification receipt", "execution recovery packet", "after-action learning packet"]:
                        if not any(str(item).startswith(command) for item in recovery_commands):
                            raise SystemExit(f"jarvis doctor recovery queue missed {command}: {metadata}")
                elif metadata.get("recent_approval_held_runs", 0) > 0:
                    if metadata.get("execution_health_recovery_closure_blocks_completion_claim") is not False:
                        raise SystemExit(f"jarvis doctor should keep approval-held-only rows out of recovery closure: {metadata}")
                    if recovery_commands:
                        raise SystemExit(f"jarvis doctor should not request recovery proof for approval-held-only rows: {metadata}")
                if metadata.get("completion_claim_ready"):
                    raise SystemExit(f"jarvis doctor should block completion while approvals/failures remain: {metadata}")
                if metadata.get("completion_blocker_count", 0) < 1 or not metadata.get("completion_blockers"):
                    raise SystemExit(f"jarvis doctor missed completion blockers: {metadata}")
                if metadata.get("recent_failed_runs", 0) > 0 and not any("recovery closure" in blocker for blocker in metadata.get("completion_blockers", [])):
                    raise SystemExit(f"jarvis doctor completion blockers missed recovery closure: {metadata}")
                if metadata.get("recent_approval_held_runs", 0) > 0 and not any("approval-held" in blocker for blocker in metadata.get("completion_blockers", [])):
                    raise SystemExit(f"jarvis doctor completion blockers missed approval-held rows: {metadata}")
                completion_queue = metadata.get("completion_proof_queue") or []
                approval_indexes = [
                    index for index, command in enumerate(completion_queue)
                    if str(command).startswith("approval readiness")
                ]
                after_action_indexes = [
                    index for index, command in enumerate(completion_queue)
                    if str(command).startswith("after-action learning packet")
                ]
                if metadata.get("pending_approvals") and approval_indexes and after_action_indexes and approval_indexes[0] > after_action_indexes[0]:
                    raise SystemExit(f"jarvis doctor completion proof queue should put approval review before learning evidence: {metadata}")
                original_list_sessions = runtime.store.list_sessions

                def failing_doctor_list_sessions(limit: int = 50):
                    raise OSError("sqlite denied near /\x55sers/example/private/doctor.sqlite")

                runtime.store.list_sessions = failing_doctor_list_sessions
                try:
                    failed_doctor = runtime.registry.get("jarvis_doctor").handler({})
                finally:
                    runtime.store.list_sessions = original_list_sessions
                if not failed_doctor.ok or "sqlite store: needs attention | unavailable" not in failed_doctor.output:
                    raise SystemExit(f"jarvis doctor SQLite failure missed stable guidance: {failed_doctor.output}")
                for expected in [
                    "JARVIS_DATA_DIR",
                    "JARVIS_DB_PATH",
                    BOOTSTRAP_CHECK_COMMAND,
                    "then retry",
                ]:
                    if expected not in failed_doctor.output:
                        raise SystemExit(f"jarvis doctor SQLite failure missed actionable guidance {expected}: {failed_doctor.output}")
                if "/\x55sers/example/private/doctor.sqlite" in failed_doctor.output or "sqlite denied" in failed_doctor.output:
                    raise SystemExit(f"jarvis doctor leaked raw SQLite exception text: {failed_doctor.output}")
                failed_metadata = failed_doctor.metadata
                if failed_metadata.get("sqlite_store_available") is not False or failed_metadata.get("sqlite_store_exception_type") != "OSError":
                    raise SystemExit(f"jarvis doctor missed SQLite failure diagnostics: {failed_metadata}")
                if "sqlite store" not in failed_metadata.get("missing", []):
                    raise SystemExit(f"jarvis doctor missing checks should include sqlite store: {failed_metadata}")
                assert_safe_read_only_metadata(failed_metadata, "failed jarvis doctor")
                assert_doctor_handoff(failed_metadata, "failed jarvis doctor")

                import jarvis_v2.tools.doctor as doctor_module

                original_doctor_subprocess_run = doctor_module.subprocess.run

                class PathBearingProbeResult:
                    returncode = 1
                    stdout = ""
                    stderr = "doctor probe detail near /var/folders/zc/jarvis-doctor-probe.log and /tmp/jarvis-doctor-probe.log"

                def path_bearing_probe_failure(*args, **kwargs):
                    return PathBearingProbeResult()

                doctor_module.subprocess.run = path_bearing_probe_failure
                try:
                    failed_probe_doctor = runtime.registry.get("jarvis_doctor").handler({})
                finally:
                    doctor_module.subprocess.run = original_doctor_subprocess_run
                if not failed_probe_doctor.ok:
                    raise SystemExit(f"jarvis doctor should still render when command probes fail: {failed_probe_doctor.output}")
                if any(fragment in failed_probe_doctor.output for fragment in ["/var/folders/", "/tmp/"]):
                    raise SystemExit(f"jarvis doctor command probe output leaked raw local path details: {failed_probe_doctor.output}")
                if "<local-path>" not in failed_probe_doctor.output:
                    raise SystemExit(f"jarvis doctor command probe output should retain redacted path marker: {failed_probe_doctor.output}")
                assert_safe_read_only_metadata(failed_probe_doctor.metadata, "failed probe jarvis doctor")
                assert_doctor_handoff(failed_probe_doctor.metadata, "failed probe jarvis doctor")


if __name__ == "__main__":
    main()
