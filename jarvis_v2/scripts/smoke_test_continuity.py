from __future__ import annotations

import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory

from jarvis_v2.agent.planner import RuleBasedPlanner
from jarvis_v2.scripts.test_runtime import make_temp_runtime
from jarvis_v2.memory.store import GoalRecord, MemoryRecord, TaskRecord


def test_planner_routes_continuity_aliases() -> None:
    p = RuleBasedPlanner()
    # Real gap found live 2026-07-09: "give me a handoff" fell through to
    # chat while bare "handoff brief" worked.
    for q in ("handoff brief", "give me a handoff"):
        actions = p.plan(q).actions
        if [a.tool_name for a in actions] != ["handoff_brief"]:
            raise SystemExit(f"handoff_brief route missed: {q!r} -> {[a.tool_name for a in actions]}")
    # Real gaps found live 2026-07-09: "show activity digest" / "show recent
    # activity" / "what changed recently" all fell through to chat while
    # bare "activity digest" / "recent activity" / "what changed" worked.
    for q in ("activity digest", "show activity digest", "recent activity", "show recent activity", "what changed", "what changed recently"):
        actions = p.plan(q).actions
        if [a.tool_name for a in actions] != ["activity_digest"]:
            raise SystemExit(f"activity_digest route missed: {q!r} -> {[a.tool_name for a in actions]}")
from jarvis_v2.tools.continuity import (
    _approval_boundary_rows_for_risky_work,
    _metadata_bool,
    _recovery_execution_readiness_token_sha256,
    _risky_next_step_approval_boundary_token_sha256,
    _risky_recovery_step_approval_boundary_token_sha256,
    make_continuity_tools,
)


READ_ONLY_FLAGS = [
    "calls_model",
    "executes_tools",
    "reads_private_data",
    "reads_personal_data",
    "reads_note_contents",
    "writes_files",
    "writes_database",
    "writes_memory",
    "writes_notes",
    "external_side_effect",
    "queues_approval",
    "requires_approval",
    "controls_computer",
    "speaks",
    "completes_tasks",
]


class HostileRow:
    def __init__(self, marker: str) -> None:
        self.marker = marker

    def keys(self):
        raise RuntimeError(self.marker)

    def __getitem__(self, _key):
        raise RuntimeError(self.marker)

    def __str__(self) -> str:
        return self.marker


def assert_continuity_metadata_bool_is_exact() -> None:
    if _metadata_bool(True) is not True or _metadata_bool(False) is not False:
        raise SystemExit("Continuity metadata bool helper should preserve exact bools.")
    for value in ("true", "false", "yes", "0", 1, 0, ["true"], {"value": True}, None):
        if _metadata_bool(value):
            raise SystemExit(f"Continuity metadata bool helper accepted malformed truthy value: {value!r}")
    for value in ("false", 0, None):
        if _metadata_bool(value, default=True) is not True:
            raise SystemExit(f"Continuity metadata bool helper should preserve conservative default for malformed value: {value!r}")


def assert_continuity_token_fingerprints_use_exact_bools() -> None:
    source = Path(__file__).resolve().parents[1] / "tools" / "continuity.py"
    text = source.read_text(encoding="utf-8")
    if "str(bool(row.get(" in text:
        raise SystemExit("Continuity row token fingerprints should use exact metadata bools, not truthy coercion.")

    proposed_step_sha256 = "a" * 64
    proposed_verification_sha256 = "b" * 64
    clean_next_rows = _approval_boundary_rows_for_risky_work(
        risk_signals=[],
        step_sha256=proposed_step_sha256,
        verification_sha256=proposed_verification_sha256,
        proof_queue=[],
        required_field="required_before_risky_next_step",
    )
    malformed_next_rows = [dict(row) for row in clean_next_rows]
    malformed_next_rows[0]["authorizes_tool_execution"] = "true"
    true_next_rows = [dict(row) for row in clean_next_rows]
    true_next_rows[0]["authorizes_tool_execution"] = True
    clean_next_token = _risky_next_step_approval_boundary_token_sha256(
        proposed_step_sha256=proposed_step_sha256,
        proposed_verification_sha256=proposed_verification_sha256,
        risk_signals=[],
        approval_proof_queue=[],
        approval_boundary_rows=clean_next_rows,
    )
    malformed_next_token = _risky_next_step_approval_boundary_token_sha256(
        proposed_step_sha256=proposed_step_sha256,
        proposed_verification_sha256=proposed_verification_sha256,
        risk_signals=[],
        approval_proof_queue=[],
        approval_boundary_rows=malformed_next_rows,
    )
    true_next_token = _risky_next_step_approval_boundary_token_sha256(
        proposed_step_sha256=proposed_step_sha256,
        proposed_verification_sha256=proposed_verification_sha256,
        risk_signals=[],
        approval_proof_queue=[],
        approval_boundary_rows=true_next_rows,
    )
    if malformed_next_token != clean_next_token:
        raise SystemExit("Malformed next-step authority strings should hash as conservative false.")
    if true_next_token == clean_next_token:
        raise SystemExit("Exact true next-step authority bool should still alter the token fingerprint.")

    clean_recovery_rows = _approval_boundary_rows_for_risky_work(
        risk_signals=[],
        step_sha256=proposed_step_sha256,
        verification_sha256=proposed_verification_sha256,
        proof_queue=[],
        required_field="required_before_risky_recovery_step",
    )
    malformed_recovery_rows = [dict(row) for row in clean_recovery_rows]
    malformed_recovery_rows[0]["authorizes_external_side_effect"] = "true"
    true_recovery_rows = [dict(row) for row in clean_recovery_rows]
    true_recovery_rows[0]["authorizes_external_side_effect"] = True
    clean_recovery_token = _risky_recovery_step_approval_boundary_token_sha256(
        recovery_step_sha256=proposed_step_sha256,
        recovery_verification_sha256=proposed_verification_sha256,
        risk_signals=[],
        approval_proof_queue=[],
        approval_boundary_rows=clean_recovery_rows,
    )
    malformed_recovery_token = _risky_recovery_step_approval_boundary_token_sha256(
        recovery_step_sha256=proposed_step_sha256,
        recovery_verification_sha256=proposed_verification_sha256,
        risk_signals=[],
        approval_proof_queue=[],
        approval_boundary_rows=malformed_recovery_rows,
    )
    true_recovery_token = _risky_recovery_step_approval_boundary_token_sha256(
        recovery_step_sha256=proposed_step_sha256,
        recovery_verification_sha256=proposed_verification_sha256,
        risk_signals=[],
        approval_proof_queue=[],
        approval_boundary_rows=true_recovery_rows,
    )
    if malformed_recovery_token != clean_recovery_token:
        raise SystemExit("Malformed recovery-boundary authority strings should hash as conservative false.")
    if true_recovery_token == clean_recovery_token:
        raise SystemExit("Exact true recovery-boundary authority bool should still alter the token fingerprint.")

    clean_scorecard = [
        {
            "item": "receipt",
            "points": 1,
            "max_points": 1,
            "ready": True,
            "required_before_normal_followthrough": True,
            "authorizes_action_now": False,
            "authorizes_risky_work": False,
            "authorizes_unreviewed_followthrough": False,
            "authorizes_model_call": False,
            "authorizes_tool_execution": False,
            "authorizes_personal_data_read": False,
            "authorizes_external_side_effect": False,
        }
    ]
    malformed_scorecard = [dict(clean_scorecard[0], authorizes_model_call="true")]
    true_scorecard = [dict(clean_scorecard[0], authorizes_model_call=True)]
    readiness_kwargs = {
        "objective": "continue safely",
        "reviewed_step": "local-safe review",
        "verification": "smoke",
        "receipt_sha256": "c" * 64,
        "receipt_file_sha256": "c" * 64,
        "receipt_hash_matches_file": True,
        "checkpoint_sha256": "d" * 64,
        "checkpoint_file_sha256": "d" * 64,
        "checkpoint_hash_matches_file": True,
        "recovery_step_approval_boundary_token_sha256": "e" * 64,
        "recovery_followthrough_token_sha256": "f" * 64,
        "local_safe_recovery_execution_token_sha256": "1" * 64,
        "recovery_execution_contract_fields": ["receipt"],
        "stop_condition": "stop on operator request",
        "risk_signals": [],
        "approval_reference_present": False,
    }
    clean_readiness_token = _recovery_execution_readiness_token_sha256(
        recovery_execution_scorecard_rows=clean_scorecard,
        **readiness_kwargs,
    )
    malformed_readiness_token = _recovery_execution_readiness_token_sha256(
        recovery_execution_scorecard_rows=malformed_scorecard,
        **readiness_kwargs,
    )
    true_readiness_token = _recovery_execution_readiness_token_sha256(
        recovery_execution_scorecard_rows=true_scorecard,
        **readiness_kwargs,
    )
    if malformed_readiness_token != clean_readiness_token:
        raise SystemExit("Malformed recovery-execution authority strings should hash as conservative false.")
    if true_readiness_token == clean_readiness_token:
        raise SystemExit("Exact true recovery-execution authority bool should still alter the token fingerprint.")


def file_sha256(path: object) -> str:
    return hashlib.sha256(Path(str(path)).read_bytes()).hexdigest()


def assert_agi_continuity_handoff(result, label: str) -> None:
    metadata = result.metadata
    required_keys = [
        "agi_next_gate",
        "agi_next_target_title",
        "agi_next_build_command",
        "agi_next_evidence_closure_commands",
        "agi_next_evidence_closure_command_count",
        "agi_next_focused_verification_commands",
        "agi_next_focused_verification_command_count",
        "agi_next_likely_files",
        "agi_next_likely_file_count",
        "agi_next_target_file_integrity_status",
        "agi_next_target_files_checked",
        "agi_next_target_files_exist",
        "agi_next_missing_target_files",
        "agi_next_missing_target_file_count",
        "agi_next_target_integrity_blocks_start",
        "agi_next_acceptance_checks",
        "agi_next_acceptance_check_count",
        "agi_next_build_packet_ready_for_review",
        "agi_focus_selection_source",
        "agi_focus_selection_reason",
        "agi_focus_canonical_selector_command",
        "agi_focus_deliberate_focus_override",
    ]
    for key in required_keys:
        if key not in metadata:
            raise SystemExit(f"{label} missed AGI handoff metadata {key}: {metadata}")
    if "AGI build-readiness handoff:" not in result.output:
        raise SystemExit(f"{label} missed AGI handoff output section: {result.output}")
    if "`agi next build move: personal integrations`" not in result.output:
        raise SystemExit(f"{label} missed selected AGI build command: {result.output}")
    if metadata.get("agi_next_gate") != "personal integrations":
        raise SystemExit(f"{label} selected the wrong AGI gate: {metadata}")
    if metadata.get("agi_next_build_command") != "agi next build move: personal integrations":
        raise SystemExit(f"{label} missed selected AGI build command metadata: {metadata}")
    if metadata.get("agi_focus_selection_source") != "operator_focus_handoff":
        raise SystemExit(f"{label} missed explicit AGI focus selection source: {metadata}")
    if metadata.get("agi_focus_canonical_selector_command") != "agi gates":
        raise SystemExit(f"{label} missed canonical AGI selector command: {metadata}")
    if metadata.get("agi_focus_deliberate_focus_override") is not True:
        raise SystemExit(f"{label} missed deliberate AGI focus override flag: {metadata}")
    if "operator-facing build target" not in str(metadata.get("agi_focus_selection_reason") or ""):
        raise SystemExit(f"{label} missed AGI focus selection reason: {metadata}")
    if metadata.get("agi_next_likely_file_count") != len(metadata.get("agi_next_likely_files") or []):
        raise SystemExit(f"{label} AGI likely-file count diverged: {metadata}")
    if metadata.get("agi_next_evidence_closure_command_count") != len(metadata.get("agi_next_evidence_closure_commands") or []):
        raise SystemExit(f"{label} AGI evidence-closure count diverged: {metadata}")
    if metadata.get("agi_next_focused_verification_command_count") != len(metadata.get("agi_next_focused_verification_commands") or []):
        raise SystemExit(f"{label} AGI focused verification count diverged: {metadata}")
    if metadata.get("agi_next_acceptance_check_count") != len(metadata.get("agi_next_acceptance_checks") or []):
        raise SystemExit(f"{label} AGI acceptance count diverged: {metadata}")
    if bool(metadata.get("agi_next_target_integrity_blocks_start")) is metadata.get("agi_next_target_files_exist"):
        raise SystemExit(f"{label} AGI integrity blocker should invert file existence: {metadata}")


def assert_return_brief_handoff(result, label: str, *, expected_limit: int) -> None:
    assert_continuity_brief_handoff(
        result,
        label,
        prefix="return_brief",
        expected_limit=expected_limit,
        expected_sections=["readiness", "activity", "safe_next_actions", "agi_build_readiness"],
        expected_commands=[
            "catch me up",
            "safety status",
            "safe next actions",
            "work queue",
            "next action packet",
            "priority stack",
            "focus brief",
            "build target packet",
            "agi next build move: personal integrations",
        ],
    )


def assert_continuity_brief_handoff(
    result,
    label: str,
    *,
    prefix: str,
    expected_limit: int,
    expected_sections: list[str],
    expected_commands: list[str],
) -> None:
    metadata = result.metadata
    handoff = metadata.get(f"{prefix}_handoff")
    if not isinstance(handoff, dict):
        raise SystemExit(f"{label} missed {prefix}_handoff: {metadata}")
    if handoff.get("source") != prefix:
        raise SystemExit(f"{label} handoff source diverged: {metadata}")
    if metadata.get(f"{prefix}_handoff_ready") is not True or handoff.get("handoff_ready") is not True:
        raise SystemExit(f"{label} should report handoff ready: {metadata}")
    if metadata.get(f"{prefix}_ready_for_operator") is not True or handoff.get("ready_for_operator") is not True:
        raise SystemExit(f"{label} should report ready_for_operator: {metadata}")
    if metadata.get(f"{prefix}_state_changed") is not False or handoff.get("state_changed") is not False:
        raise SystemExit(f"{label} handoff should report unchanged state: {metadata}")
    if metadata.get(f"{prefix}_changed") != [] or handoff.get("changed") != []:
        raise SystemExit(f"{label} handoff should report no changed resources: {metadata}")
    if metadata.get(f"{prefix}_content_in_handoff") is not False or handoff.get("content_in_handoff") is not False:
        raise SystemExit(f"{label} handoff should stay content-free: {metadata}")
    if handoff.get("limit") != expected_limit:
        raise SystemExit(f"{label} handoff limit diverged: {metadata}")
    if metadata.get(f"{prefix}_sections") != handoff.get("sections"):
        raise SystemExit(f"{label} handoff sections diverged: {metadata}")
    if metadata.get(f"{prefix}_section_count") != len(handoff.get("sections", [])):
        raise SystemExit(f"{label} handoff section count diverged: {metadata}")
    section_names = [section.get("section") for section in handoff.get("sections", [])]
    for section in expected_sections:
        if section not in section_names:
            raise SystemExit(f"{label} handoff missed section {section!r}: {metadata}")
    if metadata.get(f"{prefix}_next_commands") != handoff.get("next_commands"):
        raise SystemExit(f"{label} handoff next commands diverged: {metadata}")
    if metadata.get(f"{prefix}_next_command_count") != len(handoff.get("next_commands", [])):
        raise SystemExit(f"{label} handoff next-command count diverged: {metadata}")
    for command in expected_commands:
        if command not in handoff.get("next_commands", []):
            raise SystemExit(f"{label} handoff missed next command {command!r}: {metadata}")
    if "next_actions" in metadata and handoff.get("pending_approvals") != metadata.get("next_actions", {}).get("pending_approvals"):
        raise SystemExit(f"{label} handoff pending approval count diverged: {metadata}")
    if "next_actions" in metadata and handoff.get("open_tasks") != metadata.get("next_actions", {}).get("open_tasks"):
        raise SystemExit(f"{label} handoff open task count diverged: {metadata}")
    if "next_actions" in metadata and handoff.get("active_goals") != metadata.get("next_actions", {}).get("active_goals"):
        raise SystemExit(f"{label} handoff active goal count diverged: {metadata}")
    if handoff.get("agi_next_build_command") != metadata.get("agi_next_build_command"):
        raise SystemExit(f"{label} handoff AGI command diverged: {metadata}")
    boundaries = handoff.get("boundaries") or {}
    if metadata.get(f"{prefix}_boundaries") != boundaries:
        raise SystemExit(f"{label} handoff boundaries diverged: {metadata}")
    if boundaries.get("read_only") is not True:
        raise SystemExit(f"{label} handoff boundary should report read_only=True: {metadata}")
    for key in [
        "calls_model",
        "executes_tools",
        "queues_approval",
        "writes_files",
        "writes_memory",
        "writes_notes",
        "controls_computer",
        "authorizes_execution",
        "authorizes_completion_claim",
        "approval_granted",
    ]:
        if boundaries.get(key) is not False:
            raise SystemExit(f"{label} handoff boundary should report {key}=False: {metadata}")
    for forbidden in ["/\x55sers/", "/private/", "/var/folders/", "/tmp/"]:
        if forbidden in str(handoff):
            raise SystemExit(f"{label} handoff leaked local path fragment {forbidden!r}: {metadata}")


def assert_build_proof_queue(metadata: dict, label: str) -> None:
    queue = metadata.get("build_proof_queue") or []
    if not queue:
        raise SystemExit(f"{label} missed build proof queue: {metadata}")
    if metadata.get("build_proof_queue_count") != len(queue):
        raise SystemExit(f"{label} build proof queue count diverged: {metadata}")
    if metadata.get("build_next_required_command") != queue[0]:
        raise SystemExit(f"{label} missed first build required command: {metadata}")
    if metadata.get("build_next_proof_command") != queue[0]:
        raise SystemExit(f"{label} missed first build proof command: {metadata}")
    expected_commands = [
        "approval readiness",
        "approval packet",
        "approval chain proof",
        "build delta",
        "work block checkpoint",
        "harness completion",
        "completion claim gate",
    ]
    if metadata.get("failed_runs", 0) > 0:
        expected_commands.append("recovery closure checklist")
    for command in expected_commands:
        if not any(str(item).startswith(command) for item in queue):
            raise SystemExit(f"{label} build proof queue missed {command}: {metadata}")


def assert_checkpoint_recovery_proof_queue(metadata: dict, label: str) -> None:
    queue = metadata.get("checkpoint_recovery_proof_queue") or []
    if not queue:
        raise SystemExit(f"{label} missed checkpoint recovery proof queue: {metadata}")
    if metadata.get("checkpoint_recovery_proof_queue_count") != len(queue):
        raise SystemExit(f"{label} checkpoint recovery proof queue count diverged: {metadata}")
    if metadata.get("checkpoint_recovery_next_proof_command") != queue[0]:
        raise SystemExit(f"{label} checkpoint recovery next proof command diverged: {metadata}")
    if metadata.get("checkpoint_recovery_next_required_command") != queue[0]:
        raise SystemExit(f"{label} checkpoint recovery next required command diverged: {metadata}")
    if metadata.get("next_required_command") and metadata.get("next_required_command") != queue[0]:
        raise SystemExit(f"{label} generic next required command diverged: {metadata}")
    expected_commands = [
        "work block checkpoint",
        "approval readiness",
        "approval packet",
        "approval chain proof",
        "build delta",
    ]
    if metadata.get("failed_runs", 0) > 0:
        expected_commands.append("recovery closure checklist")
    for command in expected_commands:
        if not any(str(item).startswith(command) for item in queue):
            raise SystemExit(f"{label} checkpoint recovery proof queue missed {command}: {metadata}")


def assert_checkpoint_recovery_command_first_output(response: str, label: str) -> None:
    if "- next proof command:" in response:
        raise SystemExit(f"{label} should render next required command, not next proof command.")
    if "- next required command:" not in response:
        raise SystemExit(f"{label} missed next required command output: {response}")


def assert_vault_relative_receipt(result, root: Path, prefix: str, label: str) -> None:
    metadata = result.tool_results[0].metadata if hasattr(result, "tool_results") else result.metadata
    response = result.response if hasattr(result, "response") else result.output
    path_text = str(metadata.get("path") or "")
    path_display = metadata.get("path_display")
    receipt_line = response.split("\n", 1)[0]
    if not path_text or not Path(path_text).exists():
        raise SystemExit(f"{label} should preserve exact saved path metadata: {metadata}")
    if path_text in receipt_line:
        raise SystemExit(f"{label} should not print the raw local note path.")
    if str(root) in receipt_line or "/private/" in receipt_line or "/\x55sers/" in receipt_line:
        raise SystemExit(f"{label} receipt should not expose local temp or user paths.")
    if not isinstance(path_display, str) or not path_display.startswith(prefix):
        raise SystemExit(f"{label} missed vault-relative display metadata: {metadata}")
    if path_display not in receipt_line:
        raise SystemExit(f"{label} should print the vault-relative saved-note label.")


def assert_continuity_packets_tolerate_malformed_recent_runs(
    *,
    store,
    activity_digest,
    build_progress_report,
    build_delta_report,
    work_block_checkpoint,
    checkpoint_recovery_preview,
) -> None:
    marker = "SHOULD_NOT_LEAK_CONTINUITY_HOSTILE_RUN"
    readable_rows = store.recent_tool_runs(12)
    original_recent_tool_runs = store.recent_tool_runs

    def hostile_recent_tool_runs(limit: int = 25):
        return [HostileRow(marker), *readable_rows]

    try:
        store.recent_tool_runs = hostile_recent_tool_runs  # type: ignore[method-assign]
        results = [
            ("activity_digest", activity_digest({"limit": 12})),
            ("build_progress_report", build_progress_report({"limit": 12})),
            ("build_delta_report", build_delta_report({"limit": 12})),
            ("work_block_checkpoint", work_block_checkpoint({"limit": 12, "objective": "continue Jarvis safely"})),
            ("checkpoint_recovery_preview", checkpoint_recovery_preview({"objective": "continue Jarvis safely"})),
        ]
    finally:
        store.recent_tool_runs = original_recent_tool_runs  # type: ignore[method-assign]

    for label, result in results:
        if not result.ok:
            raise SystemExit(f"{label} should tolerate malformed recent tool-run rows: {result}")
        metadata = result.metadata
        if metadata.get("readable_tool_run_rows") != len(readable_rows):
            raise SystemExit(f"{label} missed readable recent tool-run count: {metadata}")
        if metadata.get("unreadable_tool_run_rows") != 1:
            raise SystemExit(f"{label} missed unreadable recent tool-run count: {metadata}")
        if "unreadable tool run row(s) hidden for safety" not in result.output:
            raise SystemExit(f"{label} should report hidden unreadable recent tool-run rows: {result.output}")
        payload = result.output + repr(metadata)
        if marker in payload:
            raise SystemExit(f"{label} leaked raw malformed recent tool-run row text.")
        allowed_true = {"reads_note_contents"} if label == "checkpoint_recovery_preview" else set()
        for key in [flag for flag in READ_ONLY_FLAGS if flag not in allowed_true]:
            if metadata.get(key) is not False:
                raise SystemExit(f"{label} malformed-row path should remain read-only for {key}: {metadata}")


def assert_activity_digest_separates_approval_held_recent_runs() -> None:
    with TemporaryDirectory(prefix="jarvis-continuity-approval-held-") as temp:
        runtime = make_temp_runtime(Path(temp))
        activity_digest, *_ = make_continuity_tools(runtime.store, runtime.vault)
        runtime.store.log_tool_run(
            "continuity-approval-held",
            "list_tools",
            "READ_ONLY",
            True,
            True,
            "listed tools",
            metadata={},
        )
        runtime.store.log_tool_run(
            "continuity-approval-held",
            "send_kakao",
            "HIGH_RISK",
            False,
            True,
            "transport timed out",
            metadata={"failure_stage": "transport_timeout"},
        )
        runtime.store.log_tool_run(
            "continuity-approval-held",
            "send_telegram",
            "HIGH_RISK",
            False,
            False,
            "blocked before execution; queued as approval #4",
            approval_id=4,
            metadata={"failure_kind": "approval-gate"},
        )

        result = activity_digest({"limit": 6})
        if not result.ok:
            raise SystemExit(f"activity_digest approval-held fixture failed: {result.output}")
        for expected in [
            "3 recent run(s)",
            "1 ok, 1 failed, 1 approval-held",
            "Approval-held send_telegram",
            "safety gate held before execution",
            "approval readiness 4",
            "approval packet 4",
            "Failed send_kakao",
            "transport timed out",
        ]:
            if expected not in result.output:
                raise SystemExit(f"activity_digest missed approval-held fixture text {expected!r}: {result.output}")
        for forbidden in ["approval-gate", "blocked before execution; queued as approval #4"]:
            if forbidden in result.output:
                raise SystemExit(f"activity_digest should not expose raw approval-held audit text {forbidden!r}: {result.output}")
        expected_counts = {
            "tool_runs": 3,
            "recent_ok_tool_runs": 1,
            "recent_failed_tool_runs": 1,
            "recent_approval_held_tool_runs": 1,
            "readable_tool_run_rows": 3,
            "unreadable_tool_run_rows": 0,
        }
        for key, expected in expected_counts.items():
            if result.metadata.get(key) != expected:
                raise SystemExit(f"activity_digest missed {key}={expected}: {result.metadata}")
        for key in READ_ONLY_FLAGS:
            if result.metadata.get(key) is not False:
                raise SystemExit(f"activity_digest approval-held fixture should remain read-only for {key}: {result.metadata}")


def assert_build_checkpoint_surfaces_separate_approval_held_recent_runs() -> None:
    with TemporaryDirectory(prefix="jarvis-continuity-build-approval-held-") as temp:
        runtime = make_temp_runtime(Path(temp))
        (
            _activity_digest,
            _operator_instruction_supersession_packet,
            _operator_timebox_contract,
            build_progress_report,
            build_delta_report,
            work_block_checkpoint,
            _save_build_progress,
            _save_build_delta,
            _save_work_block_checkpoint,
            checkpoint_recovery_preview,
            *_,
        ) = make_continuity_tools(runtime.store, runtime.vault)
        runtime.store.log_tool_run(
            "continuity-build-approval-held",
            "list_tools",
            "READ_ONLY",
            True,
            True,
            "listed tools",
            metadata={},
        )
        failed_run_id = runtime.store.log_tool_run(
            "continuity-build-approval-held",
            "send_kakao",
            "HIGH_RISK",
            False,
            True,
            "transport timed out",
            metadata={"failure_stage": "transport_timeout"},
        )
        held_run_id = runtime.store.log_tool_run(
            "continuity-build-approval-held",
            "send_telegram",
            "HIGH_RISK",
            False,
            False,
            "raw approval held output should stay private",
            approval_id=4,
            metadata={"failure_kind": "approval-gate", "requires_confirmation": True},
        )

        cases = [
            ("build_progress_report", build_progress_report({"limit": 6})),
            ("build_delta_report", build_delta_report({"limit": 6})),
            (
                "work_block_checkpoint",
                work_block_checkpoint({"limit": 6, "objective": "continue Jarvis safely"}),
            ),
            (
                "checkpoint_recovery_preview",
                checkpoint_recovery_preview({"objective": "continue Jarvis safely"}),
            ),
        ]
        for label, result in cases:
            if not result.ok:
                raise SystemExit(f"{label} approval-held fixture failed: {result.output}")
            metadata = result.metadata
            expected_counts = {
                "recent_failed_tool_runs": 1,
                "recent_approval_held_tool_runs": 1,
                "readable_tool_run_rows": 3,
                "unreadable_tool_run_rows": 0,
            }
            for key, expected in expected_counts.items():
                if metadata.get(key) != expected:
                    raise SystemExit(f"{label} missed {key}={expected}: {metadata}")
            if "approval_held_runs" in metadata and metadata.get("approval_held_runs") != 1:
                raise SystemExit(f"{label} missed approval_held_runs=1: {metadata}")
            payload = result.output + repr(metadata)
            for forbidden in [
                "raw approval held output should stay private",
                "failure_kind",
                "requires_confirmation",
                f"execution recovery packet {held_run_id}",
                f"verification receipt {held_run_id}",
                f"after-action learning packet {held_run_id}",
                f"execution learning closure {held_run_id}",
            ]:
                if forbidden in payload:
                    raise SystemExit(f"{label} leaked or misrouted approval-held fixture text {forbidden!r}: {payload}")
            for expected in [
                "approval readiness 4",
                "approval packet 4",
                "approval chain proof 4",
                "safety gate held before execution",
            ]:
                if expected not in payload:
                    raise SystemExit(f"{label} missed approval-held review text {expected!r}: {payload}")
            proof_queue = metadata.get("build_proof_queue") or metadata.get("checkpoint_recovery_proof_queue") or []
            for expected in [
                "approval readiness 4",
                "approval packet 4",
                "approval chain proof 4",
                "verification receipt <approved run id from approval chain proof 4>",
            ]:
                if expected not in proof_queue:
                    raise SystemExit(f"{label} missed approval-held proof command {expected!r}: {metadata}")
            if f"execution recovery packet {failed_run_id}" not in proof_queue:
                raise SystemExit(f"{label} missed true-failure recovery packet: {metadata}")
            for key in READ_ONLY_FLAGS:
                if metadata.get(key) is not False:
                    raise SystemExit(f"{label} approval-held fixture should remain read-only for {key}: {metadata}")

        text_expectations = {
            "build_progress_report": [
                "failed or blocked runs: 1",
                "approval-held runs: 1",
                "recent approval-held runs: 1",
            ],
            "build_delta_report": [
                "blocked or failed runs: 1",
                "approval-held runs: 1",
                "approval held",
            ],
            "work_block_checkpoint": [
                f"approval-held run #{held_run_id}",
                "review: `approval readiness 4`",
            ],
            "checkpoint_recovery_preview": [
                "approval-held send_telegram",
                "review: `approval readiness 4`",
            ],
        }
        result_by_label = {label: result for label, result in cases}
        for label, expected_texts in text_expectations.items():
            output = result_by_label[label].output
            for expected in expected_texts:
                if expected not in output:
                    raise SystemExit(f"{label} missed output text {expected!r}: {output}")


def assert_build_delta_tolerates_malformed_recent_messages(*, store, session_id: str, build_delta_report) -> None:
    marker = "SHOULD_NOT_LEAK_CONTINUITY_HOSTILE_MESSAGE"
    path_marker = "/\x55sers/example/Desktop/Claude code/continuity-message-leak.md"
    store.log_message(session_id, "user", f"latest question mentions {path_marker}")
    store.log_message(session_id, "assistant", f"latest answer references {path_marker}")
    readable_rows = store.recent_messages(limit=12)
    original_recent_messages = store.recent_messages

    def hostile_recent_messages(*_args, **_kwargs):
        return [HostileRow(marker), *readable_rows]

    try:
        store.recent_messages = hostile_recent_messages  # type: ignore[method-assign]
        result = build_delta_report({"limit": 12})
    finally:
        store.recent_messages = original_recent_messages  # type: ignore[method-assign]

    if not result.ok:
        raise SystemExit(f"build_delta_report should tolerate malformed recent message rows: {result}")
    metadata = result.metadata
    if metadata.get("readable_message_rows") != len(readable_rows):
        raise SystemExit(f"build_delta_report missed readable recent message count: {metadata}")
    if metadata.get("unreadable_message_rows") != 1:
        raise SystemExit(f"build_delta_report missed unreadable recent message count: {metadata}")
    if "unreadable message row(s) hidden for safety: 1" not in result.output:
        raise SystemExit(f"build_delta_report should report hidden unreadable recent message rows: {result.output}")
    payload = result.output + repr(metadata)
    if marker in payload:
        raise SystemExit("build_delta_report leaked raw malformed recent message row text.")
    if path_marker in payload or "/\x55sers/" in result.output:
        raise SystemExit(f"build_delta_report leaked raw path-shaped recent message text: {result.output}")
    if "<local-path>" not in result.output:
        raise SystemExit(f"build_delta_report should redact path-shaped recent message text: {result.output}")
    for key in READ_ONLY_FLAGS:
        if metadata.get(key) is not False:
            raise SystemExit(f"build_delta_report malformed-message path should remain read-only for {key}: {metadata}")


def assert_continuity_packets_tolerate_malformed_sessions(*, store, activity_digest, build_progress_report) -> None:
    marker = "SHOULD_NOT_LEAK_CONTINUITY_HOSTILE_SESSION"
    path_marker = "/\x55sers/example/Desktop/Claude code/continuity-session-leak"
    store.log_message(path_marker, "user", "session id redaction proof")
    readable_rows = store.list_sessions(limit=5)
    original_list_sessions = store.list_sessions

    def hostile_list_sessions(*_args, **_kwargs):
        return [HostileRow(marker), *readable_rows]

    try:
        store.list_sessions = hostile_list_sessions  # type: ignore[method-assign]
        results = [
            ("activity_digest", activity_digest({"limit": 8})),
            ("build_progress_report", build_progress_report({"limit": 18})),
        ]
    finally:
        store.list_sessions = original_list_sessions  # type: ignore[method-assign]

    for label, result in results:
        if not result.ok:
            raise SystemExit(f"{label} should tolerate malformed session rows: {result}")
        metadata = result.metadata
        if metadata.get("readable_session_rows") != len(readable_rows):
            raise SystemExit(f"{label} missed readable session count: {metadata}")
        if metadata.get("unreadable_session_rows") != 1:
            raise SystemExit(f"{label} missed unreadable session count: {metadata}")
        if "unreadable session row(s) hidden for safety: 1" not in result.output:
            raise SystemExit(f"{label} should report hidden unreadable session rows: {result.output}")
        payload = result.output + repr(metadata)
        if marker in payload:
            raise SystemExit(f"{label} leaked raw malformed session row text.")
        if path_marker in payload or "continuity-session-leak" in payload:
            raise SystemExit(f"{label} leaked raw path-shaped session id: {result.output}")
        if "<local-path>" not in result.output:
            raise SystemExit(f"{label} should redact path-shaped session id: {result.output}")
        for key in READ_ONLY_FLAGS:
            if metadata.get(key) is not False:
                raise SystemExit(f"{label} malformed-session path should remain read-only for {key}: {metadata}")


def main() -> None:
    test_planner_routes_continuity_aliases()
    assert_continuity_metadata_bool_is_exact()
    assert_continuity_token_fingerprints_use_exact_bools()
    assert_activity_digest_separates_approval_held_recent_runs()
    assert_build_checkpoint_surfaces_separate_approval_held_recent_runs()

    with TemporaryDirectory(prefix="jarvis-continuity-") as temp:
        recovery_receipt_path = Path(temp) / "recovery-receipt.md"
        recovery_receipt_path.write_text("continuity recovery receipt\n", encoding="utf-8")
        recovery_receipt_sha256 = file_sha256(recovery_receipt_path)
        recovery_checkpoint_path = Path(temp) / "work-block-checkpoint.md"
        recovery_checkpoint_path.write_text("continuity recovery checkpoint\n", encoding="utf-8")
        recovery_checkpoint_sha256 = file_sha256(recovery_checkpoint_path)
        post_step_receipt_path = Path(temp) / "post-step-receipt.md"
        post_step_receipt_path.write_text("continuity post-step receipt\n", encoding="utf-8")
        post_step_receipt_sha256 = file_sha256(post_step_receipt_path)
        post_step_checkpoint_path = Path(temp) / "post-step-checkpoint.md"
        post_step_checkpoint_path.write_text("continuity post-step checkpoint\n", encoding="utf-8")
        post_step_checkpoint_sha256 = file_sha256(post_step_checkpoint_path)
        prior_cycle_ledger_token_sha256 = "a" * 64
        runtime = make_temp_runtime(Path(temp))
        (
            activity_digest,
            operator_instruction_supersession_packet,
            operator_timebox_contract,
            build_progress_report,
            build_delta_report,
            work_block_checkpoint,
            save_build_progress,
            save_build_delta,
            save_work_block_checkpoint,
            checkpoint_recovery_preview,
            checkpoint_recovery_apply_packet,
            checkpoint_recovery_receipt,
            checkpoint_recovery_execute,
            checkpoint_recovery_followthrough_packet,
            checkpoint_recovery_cockpit,
            autonomy_resume_gate,
            autonomy_continuation_execution_packet,
            autonomy_step_closure_packet,
            autonomy_cycle_ledger,
            recent_saved_notes,
        ) = make_continuity_tools(runtime.store, runtime.vault)
        reflections = Path(temp) / "Vault" / "Jarvis" / "Reflections"
        cases = [
            "remember that continuity summaries should catch the operator up quickly",
            "add task review activity digest priority high",
            "create goal Build continuity because return from breaks with context",
            "add step to goal 1: summarize recent Jarvis changes",
            "record decision Activity digest is read only because catch-up should be safe impact no approvals needed",
            "add person Maya relation collaborator notes checks assistant progress",
            "set preference update style to concise category communication",
            "run command python3 --version",
            "return brief",
            "save return brief",
            "handoff brief",
            "save handoff brief",
            "session closeout",
            "save session closeout",
            "recent saved notes",
            "work block checkpoint: continue Jarvis V2 safely",
            "save work block checkpoint: continue Jarvis V2 safely",
            "operator timebox: continue Jarvis stop_at=2026-06-09T08:10:00+09:00 current_time=2026-06-08T20:55:00+09:00 timezone=Asia/Seoul",
            "operator timebox: continue Jarvis stop_at=2026-06-09T08:10:00+09:00 current_time=2026-06-09T08:11:00+09:00 timezone=Asia/Seoul",
            "operator timebox: continue Jarvis",
            "checkpoint recovery: continue Jarvis V2 safely",
            "checkpoint recovery cockpit: continue Jarvis V2 safely stop_at=2026-06-09T08:10:00+09:00 current_time=2026-06-08T20:55:00+09:00 timezone=Asia/Seoul",
            "checkpoint recovery apply: continue Jarvis V2 safely",
            "checkpoint recovery receipt: continue Jarvis V2 safely",
            f"checkpoint recovery follow-through: step=reviewed local-safe continuity metadata verification=continuity smoke passed receipt={recovery_receipt_path} receipt_sha256={recovery_receipt_sha256} checkpoint={recovery_checkpoint_path} checkpoint_sha256={recovery_checkpoint_sha256} stop=stop if verification drifts blockers=none",
            f"autonomy step closure: continue Jarvis safely stop_at=2099-01-01T00:00:00+09:00 current_time=2026-06-09T03:00:00+09:00 timezone=Asia/Seoul reviewed_step=reviewed local-safe continuity metadata recovery_verification=continuity smoke passed recovery_receipt_path={recovery_receipt_path} recovery_receipt_sha256={recovery_receipt_sha256} recovery_checkpoint_path={recovery_checkpoint_path} recovery_checkpoint_sha256={recovery_checkpoint_sha256} stop_condition=stop if verification drifts blockers=none next_step=update local continuity metadata next_verification=continuity smoke passed step=update local continuity metadata post_step_verification=continuity smoke passed post_step_receipt_path={post_step_receipt_path} post_step_receipt_sha256={post_step_receipt_sha256} post_step_checkpoint_path={post_step_checkpoint_path} post_step_checkpoint_sha256={post_step_checkpoint_sha256} execution_health=execution health reviewed execution_audit=execution audit reviewed after_action_learning=learning reviewed",
            "catch me up",
            "catch me up please",
            "catch up please",
            "get me up to speed",
            "bring me up to speed please",
            "what changed while I was gone",
            "work queue",
            "help continuity",
        ]
        for case in cases:
            result = runtime.handle(case)
            status = "ok" if result.verified else "blocked"
            print(f"[{status}] {case}")
            print(result.response[:2000])
            print()
            if case in {"catch me up", "catch me up please", "catch up please", "get me up to speed", "bring me up to speed please", "what changed while I was gone"}:
                required = [
                    "Jarvis activity digest",
                    "blockers:",
                    "review activity digest",
                    "Build continuity",
                    "Activity digest is read only",
                    "continuity summaries should catch the operator up quickly",
                    "Approval #",
                    "approval packet",
                    "scheduled jobs:",
                ]
                missing = [item for item in required if item not in result.response]
                if missing:
                    raise SystemExit(f"Activity digest missing expected context: {missing}")
            if case == "return brief":
                required = [
                    "# Return Brief",
                    "Jarvis readiness report",
                    "Jarvis activity digest",
                    "Safe next actions:",
                    "ready with attention",
                    "Approval #",
                    "approval packet",
                    "review activity digest",
                    "Build continuity",
                    "Do not auto-run shell",
                ]
                missing = [item for item in required if item not in result.response]
                if missing:
                    raise SystemExit(f"Return brief missing expected context: {missing}")
            if case == "save return brief":
                assert_vault_relative_receipt(result, Path(temp), "Reflections/", case)
                saved = sorted(reflections.glob("* Return Brief.md"))
                if not saved:
                    raise SystemExit("Saved return brief note was not written.")
                note_text = saved[-1].read_text(encoding="utf-8")
                required = [
                    "# Return Brief",
                    "Jarvis readiness report",
                    "Jarvis activity digest",
                    "Safe next actions:",
                    "Approval #",
                    "approval packet",
                    "Build continuity",
                ]
                missing = [item for item in required if item not in note_text]
                if missing:
                    raise SystemExit(f"Saved return brief note missing expected context: {missing}")
            if case == "handoff brief":
                required = [
                    "# Jarvis Handoff Brief",
                    "Purpose: give the operator one safe, inspectable resume point",
                    "Jarvis readiness report",
                    "Jarvis activity digest",
                    "Jarvis build progress report",
                    "Safe next actions:",
                    "Safety reminder:",
                    "approval review",
                    "approval packet",
                    "save approval review",
                    "Do not auto-run shell",
                    "Approval #",
                    "approval packet",
                    "Build continuity",
                ]
                missing = [item for item in required if item not in result.response]
                if missing:
                    raise SystemExit(f"Handoff brief missing expected context: {missing}")
            if case == "save handoff brief":
                assert_vault_relative_receipt(result, Path(temp), "Reflections/", case)
                saved = sorted(reflections.glob("* Handoff Brief.md"))
                if not saved:
                    raise SystemExit("Saved handoff brief note was not written.")
                note_text = saved[-1].read_text(encoding="utf-8")
                required = [
                    "# Jarvis Handoff Brief",
                    "Jarvis readiness report",
                    "Jarvis activity digest",
                    "Jarvis build progress report",
                    "Safety reminder:",
                    "Approval #",
                    "approval packet",
                    "Build continuity",
                ]
                missing = [item for item in required if item not in note_text]
                if missing:
                    raise SystemExit(f"Saved handoff brief note missing expected context: {missing}")
            if case == "work queue":
                required = [
                    "Jarvis work queue:",
                    "Agent landscape research guidance",
                    "Zoey/OpenClaw/Hermes research",
                    "OpenAI Agents SDK",
                    "LangGraph",
                    "human-in-the-loop",
                    "OpenHands/Devin",
                    "persistent prompt-injection",
                    "Zoey/Lindy/Zapier",
                    "broad ungated integrations",
                    "Other Jarvis-style systems",
                    "visible control plane",
                    "not companion personas",
                    "phone reliability",
                    "operator-real evals",
                ]
                missing = [item for item in required if item not in result.response]
                if missing:
                    raise SystemExit(f"Work queue missed agent landscape research guidance: {missing}")
            if case == "session closeout":
                required = [
                    "# Session Closeout",
                    "Closeout checklist:",
                    "save session closeout",
                    "save approval review",
                    "Jarvis activity digest",
                    "Jarvis build progress report",
                    "Jarvis work queue",
                    "Approval review",
                    "Approval #",
                    "approval packet",
                    "Build continuity",
                    "review activity digest",
                    "approval-gated work stays blocked",
                ]
                missing = [item for item in required if item not in result.response]
                if missing:
                    raise SystemExit(f"Session closeout missing expected context: {missing}")
            if case == "save session closeout":
                assert_vault_relative_receipt(result, Path(temp), "Reflections/", case)
                saved = sorted(reflections.glob("* Session Closeout.md"))
                if not saved:
                    raise SystemExit("Saved session closeout note was not written.")
                note_text = saved[-1].read_text(encoding="utf-8")
                required = [
                    "# Session Closeout",
                    "Closeout checklist:",
                    "Jarvis activity digest",
                    "Jarvis build progress report",
                    "Jarvis work queue",
                    "Approval review",
                    "Approval #",
                    "approval packet",
                    "Build continuity",
                ]
                missing = [item for item in required if item not in note_text]
                if missing:
                    raise SystemExit(f"Saved session closeout note missing expected context: {missing}")
            if case == "recent saved notes":
                required = [
                    "Jarvis recent saved notes",
                    "metadata-only",
                    "Reflections/",
                    "Return Brief.md",
                    "Handoff Brief.md",
                    "Session Closeout.md",
                    "Folders scanned:",
                    "Useful follow-ups:",
                    "save build delta",
                ]
                missing = [item for item in required if item not in result.response]
                if missing:
                    raise SystemExit(f"Recent saved notes missing expected context: {missing}")
                metadata = result.tool_results[0].metadata
                if metadata.get("notes", 0) < 3:
                    raise SystemExit(f"Recent saved notes expected at least 3 notes, got {metadata}")
                for key in READ_ONLY_FLAGS:
                    if metadata.get(key) is not False:
                        raise SystemExit(f"Recent saved notes unsafe metadata {key}: {metadata}")
            if case == "work block checkpoint: continue Jarvis V2 safely":
                required = [
                    "Jarvis work-block checkpoint",
                    "resumable packet",
                    "Objective:",
                    "Progress evidence:",
                    "Verification evidence:",
                    "Open risks and blockers:",
                    "Resume packet:",
                    "next safe command",
                    "Build proof queue:",
                    "next required command",
                    "proof queue count",
                    "Boundary:",
                    "read-only checkpoint",
                    "do not continue past the operator's explicit stop time",
                ]
                missing = [item for item in required if item not in result.response]
                if missing:
                    raise SystemExit(f"Work-block checkpoint missing expected context: {missing}")
                metadata = result.tool_results[0].metadata
                if metadata.get("tool_runs", 0) < 1:
                    raise SystemExit(f"Work-block checkpoint missed recent runs: {metadata}")
                if metadata.get("operator_timeboxes_override_priority") is not True or metadata.get("stop_times_override_priority") is not True:
                    raise SystemExit(f"Work-block checkpoint missed stop-time override metadata: {metadata}")
                assert_build_proof_queue(metadata, "Work-block checkpoint")
                for key in READ_ONLY_FLAGS:
                    if metadata.get(key) is not False:
                        raise SystemExit(f"Work-block checkpoint unsafe metadata {key}: {metadata}")
            if case == "save work block checkpoint: continue Jarvis V2 safely":
                assert_vault_relative_receipt(result, Path(temp), "Reflections/", case)
                required = [
                    "Work-block checkpoint saved",
                    "Jarvis work-block checkpoint",
                    "Verification evidence:",
                    "Resume packet:",
                    "Build proof queue:",
                    "Boundary:",
                    "do not continue past the operator's explicit stop time",
                ]
                missing = [item for item in required if item not in result.response]
                if missing:
                    raise SystemExit(f"Saved work-block checkpoint missing expected context: {missing}")
                saved = sorted(reflections.glob("* Work Block Checkpoint.md"))
                if not saved:
                    raise SystemExit("Saved work-block checkpoint note was not written.")
                metadata = result.tool_results[0].metadata
                if not metadata.get("writes_files") or not metadata.get("writes_notes"):
                    raise SystemExit(f"Saved work-block checkpoint did not mark note write: {metadata}")
                if metadata.get("operator_timeboxes_override_priority") is not True or metadata.get("stop_times_override_priority") is not True:
                    raise SystemExit(f"Saved work-block checkpoint missed stop-time override metadata: {metadata}")
                for key in [flag for flag in READ_ONLY_FLAGS if flag not in {"writes_files", "writes_notes"}]:
                    if metadata.get(key) is not False:
                        raise SystemExit(f"Saved work-block checkpoint unsafe metadata {key}: {metadata}")
            if case == "operator timebox: continue Jarvis stop_at=2026-06-09T08:10:00+09:00 current_time=2026-06-08T20:55:00+09:00 timezone=Asia/Seoul":
                required = [
                    "Jarvis operator timebox contract",
                    "STOP_WINDOW_ACTIVE",
                    "can continue now: yes",
                    "minutes remaining:",
                    "explicit stop times",
                    "override priority goals",
                ]
                missing = [item for item in required if item not in result.response]
                if missing:
                    raise SystemExit(f"Active operator timebox missing expected context: {missing}")
                metadata = result.tool_results[0].metadata
                if metadata.get("timebox_state") != "STOP_WINDOW_ACTIVE" or metadata.get("can_continue_now") is not True or metadata.get("should_stop_now") is not False:
                    raise SystemExit(f"Active operator timebox metadata was wrong: {metadata}")
                if metadata.get("operator_timeboxes_override_priority") is not True or metadata.get("stop_times_override_priority") is not True:
                    raise SystemExit(f"Active operator timebox missed stop-time override metadata: {metadata}")
                for key in READ_ONLY_FLAGS:
                    if metadata.get(key) is not False:
                        raise SystemExit(f"Active operator timebox unsafe metadata {key}: {metadata}")
            if case == "operator timebox: continue Jarvis stop_at=2026-06-09T08:10:00+09:00 current_time=2026-06-09T08:11:00+09:00 timezone=Asia/Seoul":
                required = [
                    "Jarvis operator timebox contract",
                    "STOP_TIME_REACHED",
                    "can continue now: no",
                    "stop and summarize work completed",
                ]
                missing = [item for item in required if item not in result.response]
                if missing:
                    raise SystemExit(f"Expired operator timebox missing expected context: {missing}")
                metadata = result.tool_results[0].metadata
                if metadata.get("timebox_state") != "STOP_TIME_REACHED" or metadata.get("can_continue_now") is not False or metadata.get("should_stop_now") is not True:
                    raise SystemExit(f"Expired operator timebox metadata was wrong: {metadata}")
                for key in READ_ONLY_FLAGS:
                    if metadata.get(key) is not False:
                        raise SystemExit(f"Expired operator timebox unsafe metadata {key}: {metadata}")
            if case == "operator timebox: continue Jarvis":
                required = [
                    "Jarvis operator timebox contract",
                    "HELD_FOR_PARSEABLE_TIMEBOX",
                    "missing proof: stop_at",
                    "can continue now: no",
                ]
                missing = [item for item in required if item not in result.response]
                if missing:
                    raise SystemExit(f"Missing-stop operator timebox missing expected context: {missing}")
                metadata = result.tool_results[0].metadata
                if metadata.get("timebox_state") != "HELD_FOR_PARSEABLE_TIMEBOX" or "stop_at" not in metadata.get("missing_timebox_proof", []):
                    raise SystemExit(f"Missing-stop operator timebox metadata was wrong: {metadata}")
                for key in READ_ONLY_FLAGS:
                    if metadata.get(key) is not False:
                        raise SystemExit(f"Missing-stop operator timebox unsafe metadata {key}: {metadata}")
            if case == "checkpoint recovery: continue Jarvis V2 safely":
                required = [
                    "Jarvis checkpoint recovery preview",
                    "Latest saved checkpoint:",
                    "Resume-and-verify sequence:",
                    "Current blockers:",
                    "Safe next command:",
                    "Checkpoint recovery proof queue:",
                ]
                missing = [item for item in required if item not in result.response]
                if missing:
                    raise SystemExit(f"Checkpoint recovery preview missing expected context: {missing}")
                metadata = result.tool_results[0].metadata
                if not metadata.get("checkpoint_found") or not metadata.get("checkpoint_read"):
                    raise SystemExit(f"Checkpoint recovery should read the saved checkpoint: {metadata}")
                if metadata.get("operator_timeboxes_override_priority") is not True or metadata.get("stop_times_override_priority") is not True:
                    raise SystemExit(f"Checkpoint recovery missed stop-time override metadata: {metadata}")
                assert_checkpoint_recovery_proof_queue(metadata, "Checkpoint recovery")
                assert_checkpoint_recovery_command_first_output(result.response, "Checkpoint recovery")
                for key in [flag for flag in READ_ONLY_FLAGS if flag != "reads_note_contents"]:
                    if metadata.get(key) is not False:
                        raise SystemExit(f"Checkpoint recovery unsafe metadata {key}: {metadata}")
            if case.startswith("checkpoint recovery cockpit:"):
                required = [
                    "Jarvis checkpoint recovery cockpit",
                    "Cockpit state:",
                    "Operator timebox:",
                    "Checkpoint recovery:",
                    "Recovery proof queue:",
                    "Required closure before normal follow-through:",
                    "Boundary:",
                ]
                missing = [item for item in required if item not in result.response]
                if missing:
                    raise SystemExit(f"Checkpoint recovery cockpit missing expected context: {missing}")
                metadata = result.tool_results[0].metadata
                if metadata.get("cockpit_state") != "RECOVERY_COCKPIT_HELD":
                    raise SystemExit(f"Checkpoint recovery cockpit should hold with pending blockers: {metadata}")
                if metadata.get("timebox_state") != "STOP_WINDOW_ACTIVE" or metadata.get("can_continue_now") is not True:
                    raise SystemExit(f"Checkpoint recovery cockpit missed active timebox metadata: {metadata}")
                if metadata.get("can_resume_local_safe_review") is not False:
                    raise SystemExit(f"Checkpoint recovery cockpit should not resume with pending blockers: {metadata}")
                if not metadata.get("blockers") or metadata.get("blocker_count") != len(metadata.get("blockers")):
                    raise SystemExit(f"Checkpoint recovery cockpit missed blockers metadata: {metadata}")
                assert_checkpoint_recovery_proof_queue(metadata, "Checkpoint recovery cockpit")
                assert_checkpoint_recovery_command_first_output(result.response, "Checkpoint recovery cockpit")
                for key in READ_ONLY_FLAGS:
                    if metadata.get(key) is not False:
                        raise SystemExit(f"Checkpoint recovery cockpit unsafe metadata {key}: {metadata}")
            if case == "checkpoint recovery apply: continue Jarvis V2 safely":
                required = [
                    "Jarvis checkpoint recovery apply packet",
                    "approval-gated planning only",
                    "Approval-gated apply sequence:",
                    "Recovery preview excerpt:",
                    "Stop conditions:",
                    "Boundary:",
                ]
                missing = [item for item in required if item not in result.response]
                if missing:
                    raise SystemExit(f"Checkpoint recovery apply packet missing expected context: {missing}")
                metadata = result.tool_results[0].metadata
                if not metadata.get("apply_requires_approval"):
                    raise SystemExit(f"Checkpoint recovery apply should mark approval requirement: {metadata}")
                if metadata.get("operator_timeboxes_override_priority") is not True or metadata.get("stop_times_override_priority") is not True:
                    raise SystemExit(f"Checkpoint recovery apply missed stop-time override metadata: {metadata}")
                assert_checkpoint_recovery_proof_queue(metadata, "Checkpoint recovery apply")
                for key in READ_ONLY_FLAGS:
                    if metadata.get(key) is not False:
                        raise SystemExit(f"Checkpoint recovery apply unsafe metadata {key}: {metadata}")
            if case == "checkpoint recovery receipt: continue Jarvis V2 safely":
                assert_vault_relative_receipt(result, Path(temp), "Reflections/", case)
                required = [
                    "Checkpoint recovery receipt saved",
                    "Jarvis checkpoint recovery receipt",
                    "Reviewed step:",
                    "Verification evidence:",
                    "Outcome:",
                    "Boundary:",
                ]
                missing = [item for item in required if item not in result.response]
                if missing:
                    raise SystemExit(f"Checkpoint recovery receipt missing expected context: {missing}")
                saved = sorted(reflections.glob("* Checkpoint Recovery Receipt.md"))
                if not saved:
                    raise SystemExit("Checkpoint recovery receipt note was not written.")
                metadata = result.tool_results[0].metadata
                if not metadata.get("writes_files") or not metadata.get("writes_notes"):
                    raise SystemExit(f"Checkpoint recovery receipt did not mark note write: {metadata}")
                for key in [flag for flag in READ_ONLY_FLAGS if flag not in {"writes_files", "writes_notes"}]:
                    if metadata.get(key) is not False:
                        raise SystemExit(f"Checkpoint recovery receipt unsafe metadata {key}: {metadata}")
            if case.startswith("checkpoint recovery follow-through:"):
                required = [
                    "Jarvis checkpoint recovery follow-through packet",
                    "Follow-through state:",
                    "RECOVERY_FOLLOWTHROUGH_READY",
                    "normal follow-through allowed: yes",
                    "Closure evidence:",
                    "Boundary:",
                ]
                missing = [item for item in required if item not in result.response]
                if missing:
                    raise SystemExit(f"Checkpoint recovery follow-through missing expected context: {missing}")
                metadata = result.tool_results[0].metadata
                if metadata.get("followthrough_state") != "RECOVERY_FOLLOWTHROUGH_READY":
                    raise SystemExit(f"Checkpoint recovery follow-through should be ready: {metadata}")
                if metadata.get("normal_followthrough_allowed") is not True or metadata.get("recovery_closure_missing"):
                    raise SystemExit(f"Checkpoint recovery follow-through should allow normal follow-through with complete proof: {metadata}")
                for key in READ_ONLY_FLAGS:
                    if metadata.get(key) is not False:
                        raise SystemExit(f"Checkpoint recovery follow-through unsafe metadata {key}: {metadata}")
            if case.startswith("autonomy step closure:"):
                required = [
                    "Jarvis autonomy step closure packet",
                    "Closure state:",
                    "AUTONOMY_STEP_CLOSURE_HELD",
                    "ready for next continuation review: no",
                    "ready autonomy continuation execution packet",
                    "Post-step evidence:",
                    "Required commands:",
                    "Boundary:",
                ]
                missing = [item for item in required if item not in result.response]
                if missing:
                    raise SystemExit(f"Autonomy step closure missing expected context: {missing}")
                metadata = result.tool_results[0].metadata
                if metadata.get("closure_state") != "AUTONOMY_STEP_CLOSURE_HELD":
                    raise SystemExit(f"Autonomy step closure should hold while pre-step permission is blocked: {metadata}")
                if metadata.get("ready_for_next_continuation_review") is not False:
                    raise SystemExit(f"Autonomy step closure should block next continuation review without pre-step proof: {metadata}")
                if "ready autonomy continuation execution packet" not in metadata.get("missing_blockers", []):
                    raise SystemExit(f"Autonomy step closure missed pre-step blocker: {metadata}")
                for key in READ_ONLY_FLAGS:
                    if metadata.get(key) is not False:
                        raise SystemExit(f"Autonomy step closure unsafe metadata {key}: {metadata}")
            if case == "help continuity":
                for expected in [
                    "return brief",
                    "save return brief",
                    "handoff brief",
                    "save handoff brief",
                    "session closeout",
                    "save session closeout",
                    "recent saved notes",
                    "work block checkpoint",
                    "checkpoint recovery",
                    "checkpoint recovery apply",
                    "checkpoint recovery receipt",
                    "checkpoint recovery follow-through",
                    "operator timebox",
                    "autonomy step closure",
                    "save work block checkpoint",
                    "what did I miss",
                ]:
                    if expected not in result.response:
                        raise SystemExit(f"Continuity help missing: {expected}")

        digest_result = activity_digest({"limit": "not-a-number"})
        if digest_result.metadata.get("limit") != 8 or digest_result.metadata.get("writes_notes"):
            raise SystemExit("activity_digest should sanitize bad limits and remain read-only.")
        for key in READ_ONLY_FLAGS:
            if digest_result.metadata.get(key) is not False:
                raise SystemExit(f"activity_digest unsafe metadata {key}: {digest_result.metadata}")

        runtime.store.add_memory(MemoryRecord("facts", "/\x55sers/example/Desktop/Claude code/continuity-memory.md", "continuity digest path smoke"))
        runtime.store.add_task(TaskRecord("/private/tmp/continuity-task.md should not leak"))
        path_goal_id = runtime.store.create_goal(GoalRecord("/\x55sers/example/Desktop/continuity-goal.md", "continuity path smoke"))
        runtime.store.add_goal_step(path_goal_id, "/private/tmp/continuity-step.md should not leak")
        runtime.store.log_tool_run(
            runtime.session_id,
            "path_tool",
            "LOCAL_SAFE",
            False,
            False,
            "failed at /\x55sers/example/Desktop/Claude code/continuity-output.txt",
        )
        path_digest = activity_digest({"limit": 25})
        if "/\x55sers/" in path_digest.output or "/private/" in path_digest.output or "<local-path>" not in path_digest.output:
            raise SystemExit(f"activity_digest should redact path-shaped stored state: {path_digest.output}")
        for key in READ_ONLY_FLAGS:
            if path_digest.metadata.get(key) is not False:
                raise SystemExit(f"path-shaped activity_digest should remain read-only: {path_digest.metadata}")

        assert_continuity_packets_tolerate_malformed_recent_runs(
            store=runtime.store,
            activity_digest=activity_digest,
            build_progress_report=build_progress_report,
            build_delta_report=build_delta_report,
            work_block_checkpoint=work_block_checkpoint,
            checkpoint_recovery_preview=checkpoint_recovery_preview,
        )
        assert_build_delta_tolerates_malformed_recent_messages(
            store=runtime.store,
            session_id=runtime.session_id,
            build_delta_report=build_delta_report,
        )
        assert_continuity_packets_tolerate_malformed_sessions(
            store=runtime.store,
            activity_digest=activity_digest,
            build_progress_report=build_progress_report,
        )

        direct_return = runtime.registry.get("return_brief").handler({"limit": False})
        if not direct_return.ok or direct_return.metadata.get("activity", {}).get("limit") != 6:
            raise SystemExit("return_brief should treat boolean limits as malformed and use its default.")
        assert_agi_continuity_handoff(direct_return, "return_brief direct")
        assert_return_brief_handoff(direct_return, "return_brief direct", expected_limit=6)
        for key in READ_ONLY_FLAGS:
            if direct_return.metadata.get(key) is not False:
                raise SystemExit(f"return_brief direct should remain read-only for {key}: {direct_return.metadata}")

        saved_return = runtime.registry.get("save_return_brief").handler({"limit": "bad"})
        if not saved_return.ok or saved_return.metadata.get("activity", {}).get("limit") != 6 or "path" not in saved_return.metadata:
            raise SystemExit("save_return_brief should sanitize bad limits and preserve saved path metadata.")
        assert_agi_continuity_handoff(saved_return, "save_return_brief direct")
        assert_return_brief_handoff(saved_return, "save_return_brief direct", expected_limit=6)
        assert_vault_relative_receipt(saved_return, Path(temp), "Reflections/", "save_return_brief direct")
        if not saved_return.metadata.get("writes_files") or not saved_return.metadata.get("writes_notes"):
            raise SystemExit(f"save_return_brief should declare note/file writes: {saved_return.metadata}")

        direct_handoff = runtime.registry.get("handoff_brief").handler({"limit": True})
        if not direct_handoff.ok or direct_handoff.metadata.get("activity", {}).get("limit") != 8:
            raise SystemExit("handoff_brief should treat boolean limits as malformed and use its default.")
        assert_agi_continuity_handoff(direct_handoff, "handoff_brief direct")
        assert_continuity_brief_handoff(
            direct_handoff,
            "handoff_brief direct",
            prefix="handoff_brief",
            expected_limit=8,
            expected_sections=["readiness", "activity", "build_progress", "safe_next_actions", "agi_build_readiness"],
            expected_commands=["return brief", "handoff brief", "session closeout", "save handoff brief", "save session closeout", "safe next actions", "work queue", "build target packet", "agi next build move: personal integrations"],
        )
        for key in READ_ONLY_FLAGS:
            if direct_handoff.metadata.get(key) is not False:
                raise SystemExit(f"handoff_brief direct should remain read-only for {key}: {direct_handoff.metadata}")

        saved_handoff = runtime.registry.get("save_handoff_brief").handler({"limit": "bad"})
        if not saved_handoff.ok or saved_handoff.metadata.get("activity", {}).get("limit") != 8 or "path" not in saved_handoff.metadata:
            raise SystemExit("save_handoff_brief should sanitize bad limits and preserve saved path metadata.")
        assert_agi_continuity_handoff(saved_handoff, "save_handoff_brief direct")
        assert_continuity_brief_handoff(
            saved_handoff,
            "save_handoff_brief direct",
            prefix="handoff_brief",
            expected_limit=8,
            expected_sections=["readiness", "activity", "build_progress", "safe_next_actions", "agi_build_readiness"],
            expected_commands=["return brief", "handoff brief", "session closeout", "save handoff brief", "save session closeout", "safe next actions", "work queue", "build target packet", "agi next build move: personal integrations"],
        )
        assert_vault_relative_receipt(saved_handoff, Path(temp), "Reflections/", "save_handoff_brief direct")
        if not saved_handoff.metadata.get("writes_files") or not saved_handoff.metadata.get("writes_notes"):
            raise SystemExit(f"save_handoff_brief should declare note/file writes: {saved_handoff.metadata}")

        direct_closeout = runtime.registry.get("session_closeout").handler({"limit": False})
        if not direct_closeout.ok or direct_closeout.metadata.get("activity", {}).get("limit") != 8:
            raise SystemExit("session_closeout should treat boolean limits as malformed and use its default.")
        assert_agi_continuity_handoff(direct_closeout, "session_closeout direct")
        assert_continuity_brief_handoff(
            direct_closeout,
            "session_closeout direct",
            prefix="session_closeout",
            expected_limit=8,
            expected_sections=["activity", "build_progress", "work_queue", "approval_review", "agi_build_readiness"],
            expected_commands=["return brief", "handoff brief", "session closeout", "save handoff brief", "save session closeout", "approval review", "save approval review", "work queue", "build target packet", "agi next build move: personal integrations"],
        )
        for key in READ_ONLY_FLAGS:
            if direct_closeout.metadata.get(key) is not False:
                raise SystemExit(f"session_closeout direct should remain read-only for {key}: {direct_closeout.metadata}")

        saved_closeout = runtime.registry.get("save_session_closeout").handler({"limit": "bad"})
        if not saved_closeout.ok or saved_closeout.metadata.get("activity", {}).get("limit") != 8 or "path" not in saved_closeout.metadata:
            raise SystemExit("save_session_closeout should sanitize bad limits and preserve saved path metadata.")
        assert_agi_continuity_handoff(saved_closeout, "save_session_closeout direct")
        assert_continuity_brief_handoff(
            saved_closeout,
            "save_session_closeout direct",
            prefix="session_closeout",
            expected_limit=8,
            expected_sections=["activity", "build_progress", "work_queue", "approval_review", "agi_build_readiness"],
            expected_commands=["return brief", "handoff brief", "session closeout", "save handoff brief", "save session closeout", "approval review", "save approval review", "work queue", "build target packet", "agi next build move: personal integrations"],
        )
        assert_vault_relative_receipt(saved_closeout, Path(temp), "Reflections/", "save_session_closeout direct")
        if not saved_closeout.metadata.get("writes_files") or not saved_closeout.metadata.get("writes_notes"):
            raise SystemExit(f"save_session_closeout should declare note/file writes: {saved_closeout.metadata}")

        direct_active_timebox = operator_timebox_contract({
            "objective": "continue Jarvis",
            "stop_at": "2026-06-09T08:10:00+09:00",
            "current_time": "2026-06-08T20:55:00+09:00",
            "timezone": "Asia/Seoul",
        })
        if direct_active_timebox.metadata.get("timebox_state") != "STOP_WINDOW_ACTIVE" or direct_active_timebox.metadata.get("can_continue_now") is not True:
            raise SystemExit(f"direct active operator_timebox_contract failed: {direct_active_timebox.metadata}")
        for key in READ_ONLY_FLAGS:
            if direct_active_timebox.metadata.get(key) is not False:
                raise SystemExit(f"direct active operator_timebox_contract unsafe metadata {key}: {direct_active_timebox.metadata}")

        direct_expired_timebox = operator_timebox_contract({
            "objective": "continue Jarvis",
            "stop_at": "2026-06-09T08:10:00+09:00",
            "current_time": "2026-06-09T08:11:00+09:00",
            "timezone": "Asia/Seoul",
        })
        if direct_expired_timebox.metadata.get("timebox_state") != "STOP_TIME_REACHED" or direct_expired_timebox.metadata.get("should_stop_now") is not True:
            raise SystemExit(f"direct expired operator_timebox_contract failed: {direct_expired_timebox.metadata}")
        for key in READ_ONLY_FLAGS:
            if direct_expired_timebox.metadata.get(key) is not False:
                raise SystemExit(f"direct expired operator_timebox_contract unsafe metadata {key}: {direct_expired_timebox.metadata}")

        direct_missing_timebox = operator_timebox_contract({"objective": "continue Jarvis"})
        if direct_missing_timebox.metadata.get("timebox_state") != "HELD_FOR_PARSEABLE_TIMEBOX" or "stop_at" not in direct_missing_timebox.metadata.get("missing_timebox_proof", []):
            raise SystemExit(f"direct missing operator_timebox_contract failed: {direct_missing_timebox.metadata}")
        for key in READ_ONLY_FLAGS:
            if direct_missing_timebox.metadata.get(key) is not False:
                raise SystemExit(f"direct missing operator_timebox_contract unsafe metadata {key}: {direct_missing_timebox.metadata}")

        progress_result = build_progress_report({"limit": 999999})
        if progress_result.metadata.get("limit") != 200 or progress_result.metadata.get("writes_notes"):
            raise SystemExit("build_progress_report should clamp huge limits and remain read-only.")
        if "does not override the operator's explicit stop times" not in progress_result.output:
            raise SystemExit("build_progress_report missed stop-time override text.")
        if progress_result.metadata.get("operator_timeboxes_override_priority") is not True or progress_result.metadata.get("stop_times_override_priority") is not True:
            raise SystemExit(f"build_progress_report missed stop-time override metadata: {progress_result.metadata}")
        if "Build proof queue:" not in progress_result.output or "next required command" not in progress_result.output:
            raise SystemExit("build_progress_report missed build proof queue text.")
        if "next proof command" in progress_result.output:
            raise SystemExit("build_progress_report used stale proof-first queue-head prose.")
        assert_build_proof_queue(progress_result.metadata, "build_progress_report")
        for key in READ_ONLY_FLAGS:
            if progress_result.metadata.get(key) is not False:
                raise SystemExit(f"build_progress_report unsafe metadata {key}: {progress_result.metadata}")

        bool_progress = build_progress_report({"limit": True})
        if bool_progress.metadata.get("limit") != 18 or bool_progress.metadata.get("writes_notes"):
            raise SystemExit(f"build_progress_report should treat boolean limits as malformed defaults: {bool_progress.metadata}")
        for key in READ_ONLY_FLAGS:
            if bool_progress.metadata.get(key) is not False:
                raise SystemExit(f"build_progress_report boolean limit unsafe metadata {key}: {bool_progress.metadata}")

        delta_result = build_delta_report({"limit": -5})
        if delta_result.metadata.get("limit") != 1 or delta_result.metadata.get("writes_notes"):
            raise SystemExit("build_delta_report should clamp low limits and remain read-only.")
        if "Respect the operator's explicit stop times" not in delta_result.output:
            raise SystemExit("build_delta_report missed stop-time override text.")
        if delta_result.metadata.get("operator_timeboxes_override_priority") is not True or delta_result.metadata.get("stop_times_override_priority") is not True:
            raise SystemExit(f"build_delta_report missed stop-time override metadata: {delta_result.metadata}")
        if "Build proof queue:" not in delta_result.output or "next required command" not in delta_result.output:
            raise SystemExit("build_delta_report missed build proof queue text.")
        if "next proof command" in delta_result.output:
            raise SystemExit("build_delta_report used stale proof-first queue-head prose.")
        assert_build_proof_queue(delta_result.metadata, "build_delta_report")
        for key in READ_ONLY_FLAGS:
            if delta_result.metadata.get(key) is not False:
                raise SystemExit(f"build_delta_report unsafe metadata {key}: {delta_result.metadata}")

        bool_delta = build_delta_report({"limit": False})
        if bool_delta.metadata.get("limit") != 12 or bool_delta.metadata.get("writes_notes"):
            raise SystemExit(f"build_delta_report should treat boolean limits as malformed defaults: {bool_delta.metadata}")
        for key in READ_ONLY_FLAGS:
            if bool_delta.metadata.get(key) is not False:
                raise SystemExit(f"build_delta_report boolean limit unsafe metadata {key}: {bool_delta.metadata}")

        checkpoint_result = work_block_checkpoint({"limit": -5, "objective": "continue Jarvis safely"})
        if checkpoint_result.metadata.get("limit") != 1 or checkpoint_result.metadata.get("writes_notes"):
            raise SystemExit("work_block_checkpoint should clamp low limits and remain read-only.")
        if "do not continue past the operator's explicit stop time" not in checkpoint_result.output:
            raise SystemExit("work_block_checkpoint missed stop-time override text.")
        if checkpoint_result.metadata.get("operator_timeboxes_override_priority") is not True or checkpoint_result.metadata.get("stop_times_override_priority") is not True:
            raise SystemExit(f"work_block_checkpoint missed stop-time override metadata: {checkpoint_result.metadata}")
        assert_build_proof_queue(checkpoint_result.metadata, "work_block_checkpoint")
        if "Build proof queue:" not in checkpoint_result.output or "next required command" not in checkpoint_result.output:
            raise SystemExit("work_block_checkpoint missed build proof queue text.")
        if "next proof command" in checkpoint_result.output:
            raise SystemExit("work_block_checkpoint used stale proof-first queue-head prose.")
        for key in READ_ONLY_FLAGS:
            if checkpoint_result.metadata.get(key) is not False:
                raise SystemExit(f"work_block_checkpoint unsafe metadata {key}: {checkpoint_result.metadata}")

        bool_checkpoint = work_block_checkpoint({"limit": True, "objective": "continue Jarvis safely"})
        if bool_checkpoint.metadata.get("limit") != 12 or bool_checkpoint.metadata.get("writes_notes"):
            raise SystemExit(f"work_block_checkpoint should treat boolean limits as malformed defaults: {bool_checkpoint.metadata}")
        for key in READ_ONLY_FLAGS:
            if bool_checkpoint.metadata.get(key) is not False:
                raise SystemExit(f"work_block_checkpoint boolean limit unsafe metadata {key}: {bool_checkpoint.metadata}")

        saved_progress = save_build_progress({"limit": "bad"})
        if saved_progress.metadata.get("limit") != 18 or not saved_progress.metadata.get("writes_files") or not saved_progress.metadata.get("writes_notes"):
            raise SystemExit("save_build_progress should sanitize limits and mark note writes.")
        assert_vault_relative_receipt(saved_progress, Path(temp), "Reflections/", "save_build_progress direct")
        for key in [flag for flag in READ_ONLY_FLAGS if flag not in {"writes_files", "writes_notes"}]:
            if saved_progress.metadata.get(key) is not False:
                raise SystemExit(f"save_build_progress unsafe metadata {key}: {saved_progress.metadata}")

        saved_delta = save_build_delta({"limit": 50000})
        if saved_delta.metadata.get("limit") != 40 or not saved_delta.metadata.get("writes_files") or not saved_delta.metadata.get("writes_notes"):
            raise SystemExit("save_build_delta should clamp limits and mark note writes.")
        assert_vault_relative_receipt(saved_delta, Path(temp), "Reflections/", "save_build_delta direct")
        for key in [flag for flag in READ_ONLY_FLAGS if flag not in {"writes_files", "writes_notes"}]:
            if saved_delta.metadata.get(key) is not False:
                raise SystemExit(f"save_build_delta unsafe metadata {key}: {saved_delta.metadata}")

        saved_checkpoint = save_work_block_checkpoint({"limit": 50000, "objective": "continue Jarvis safely"})
        if saved_checkpoint.metadata.get("limit") != 40 or not saved_checkpoint.metadata.get("writes_files") or not saved_checkpoint.metadata.get("writes_notes"):
            raise SystemExit("save_work_block_checkpoint should clamp limits and mark note writes.")
        assert_vault_relative_receipt(saved_checkpoint, Path(temp), "Reflections/", "save_work_block_checkpoint direct")
        for key in [flag for flag in READ_ONLY_FLAGS if flag not in {"writes_files", "writes_notes"}]:
            if saved_checkpoint.metadata.get(key) is not False:
                raise SystemExit(f"save_work_block_checkpoint unsafe metadata {key}: {saved_checkpoint.metadata}")

        recovery_result = checkpoint_recovery_preview({"objective": "continue Jarvis safely"})
        if not recovery_result.metadata.get("checkpoint_found") or not recovery_result.metadata.get("checkpoint_read"):
            raise SystemExit(f"checkpoint_recovery_preview should read the saved checkpoint: {recovery_result.metadata}")
        if str(Path(temp)) in recovery_result.output or "/private/" in recovery_result.output or "/\x55sers/" in recovery_result.output:
            raise SystemExit(f"checkpoint_recovery_preview should not render absolute local checkpoint paths: {recovery_result.output}")
        if not str(recovery_result.metadata.get("latest_checkpoint_display") or "").startswith("Reflections/"):
            raise SystemExit(f"checkpoint_recovery_preview missed vault-relative checkpoint display: {recovery_result.metadata}")
        assert_checkpoint_recovery_proof_queue(recovery_result.metadata, "checkpoint_recovery_preview")
        assert_checkpoint_recovery_command_first_output(recovery_result.output, "checkpoint_recovery_preview")
        for key in [flag for flag in READ_ONLY_FLAGS if flag != "reads_note_contents"]:
            if recovery_result.metadata.get(key) is not False:
                raise SystemExit(f"checkpoint_recovery_preview unsafe metadata {key}: {recovery_result.metadata}")

        apply_result = checkpoint_recovery_apply_packet({"objective": "continue Jarvis safely"})
        if not apply_result.metadata.get("apply_requires_approval"):
            raise SystemExit(f"checkpoint_recovery_apply_packet should mark approval requirement: {apply_result.metadata}")
        assert_checkpoint_recovery_proof_queue(apply_result.metadata, "checkpoint_recovery_apply_packet")
        if apply_result.metadata.get("recovery_apply_next_required_command") != apply_result.metadata.get("recovery_apply_next_proof_command"):
            raise SystemExit(f"checkpoint_recovery_apply_packet missed recovery apply next required alias: {apply_result.metadata}")
        assert_checkpoint_recovery_command_first_output(apply_result.output, "checkpoint_recovery_apply_packet")
        for key in READ_ONLY_FLAGS:
            if apply_result.metadata.get(key) is not False:
                raise SystemExit(f"checkpoint_recovery_apply_packet unsafe metadata {key}: {apply_result.metadata}")

        cockpit_result = checkpoint_recovery_cockpit({
            "objective": "continue Jarvis safely",
            "stop_at": "2026-06-09T08:10:00+09:00",
            "current_time": "2026-06-08T20:55:00+09:00",
            "timezone": "Asia/Seoul",
        })
        if cockpit_result.metadata.get("cockpit_state") != "RECOVERY_COCKPIT_HELD":
            raise SystemExit(f"checkpoint_recovery_cockpit should hold with pending blockers: {cockpit_result.metadata}")
        if cockpit_result.metadata.get("timebox_state") != "STOP_WINDOW_ACTIVE" or cockpit_result.metadata.get("can_continue_now") is not True:
            raise SystemExit(f"checkpoint_recovery_cockpit missed timebox metadata: {cockpit_result.metadata}")
        if str(Path(temp)) in cockpit_result.output or "/private/" in cockpit_result.output or "/\x55sers/" in cockpit_result.output:
            raise SystemExit(f"checkpoint_recovery_cockpit should not render absolute local checkpoint paths: {cockpit_result.output}")
        if not str(cockpit_result.metadata.get("latest_checkpoint_display") or "").startswith("Reflections/"):
            raise SystemExit(f"checkpoint_recovery_cockpit missed vault-relative checkpoint display: {cockpit_result.metadata}")
        assert_checkpoint_recovery_proof_queue(cockpit_result.metadata, "checkpoint_recovery_cockpit")
        if cockpit_result.metadata.get("checkpoint_route_next_required_command") != cockpit_result.metadata.get("checkpoint_route_next_recovery_command"):
            raise SystemExit(f"checkpoint_recovery_cockpit missed route next required alias: {cockpit_result.metadata}")
        assert_checkpoint_recovery_command_first_output(cockpit_result.output, "checkpoint_recovery_cockpit")
        for key in READ_ONLY_FLAGS:
            if cockpit_result.metadata.get(key) is not False:
                raise SystemExit(f"checkpoint_recovery_cockpit unsafe metadata {key}: {cockpit_result.metadata}")

        receipt_result = checkpoint_recovery_receipt({
            "objective": "continue Jarvis safely",
            "approved_step": "added checkpoint receipt smoke coverage",
            "verification": "python3 -m jarvis_v2.scripts.smoke_test_continuity",
            "files": "jarvis_v2/tools/continuity.py",
            "outcome": "ready for review",
        })
        if not receipt_result.metadata.get("writes_files") or not receipt_result.metadata.get("writes_notes"):
            raise SystemExit("checkpoint_recovery_receipt should mark note writes.")
        assert_vault_relative_receipt(receipt_result, Path(temp), "Reflections/", "checkpoint_recovery_receipt direct")
        for key in [flag for flag in READ_ONLY_FLAGS if flag not in {"writes_files", "writes_notes"}]:
            if receipt_result.metadata.get(key) is not False:
                raise SystemExit(f"checkpoint_recovery_receipt unsafe metadata {key}: {receipt_result.metadata}")

        held_execute = checkpoint_recovery_execute({
            "objective": "continue Jarvis safely",
            "step": "record reviewed local-safe recovery",
        })
        if held_execute.ok or "Execution held" not in held_execute.output or "verification" not in held_execute.output:
            raise SystemExit(f"checkpoint_recovery_execute should hold until reviewed and verified: {held_execute.output}")
        if "Recovery closure gate" not in held_execute.output or "normal follow-through allowed: no" not in held_execute.output:
            raise SystemExit(f"held checkpoint_recovery_execute missed recovery closure gate: {held_execute.output}")
        if held_execute.metadata.get("recovery_followthrough_gate_state") != "HELD_BEFORE_NORMAL_FOLLOWTHROUGH":
            raise SystemExit(f"held checkpoint_recovery_execute missed held follow-through gate: {held_execute.metadata}")
        if held_execute.metadata.get("normal_followthrough_allowed") is not False or not held_execute.metadata.get("recovery_closure_missing"):
            raise SystemExit(f"held checkpoint_recovery_execute should block normal follow-through with missing proof: {held_execute.metadata}")
        for key in READ_ONLY_FLAGS:
            if held_execute.metadata.get(key) is not False:
                raise SystemExit(f"held checkpoint_recovery_execute unsafe metadata {key}: {held_execute.metadata}")

        risky_execute = checkpoint_recovery_execute({
            "objective": "continue Jarvis safely",
            "reviewed": "true",
            "step": "run command python3 --version from the recovery checkpoint",
            "verification": "verification receipt will be reviewed after approval",
            "files": "not specified",
            "outcome": "pending approval",
        })
        if risky_execute.ok or "approval_reference_for_risky_recovery_step" not in risky_execute.output:
            raise SystemExit(f"checkpoint_recovery_execute should hold risky steps without approval proof: {risky_execute.output}")
        if risky_execute.metadata.get("risky_recovery_without_approval") is not True:
            raise SystemExit(f"checkpoint_recovery_execute missed risky-without-approval metadata: {risky_execute.metadata}")
        if "shell/code" not in risky_execute.metadata.get("risky_recovery_signals", []):
            raise SystemExit(f"checkpoint_recovery_execute missed shell/code risk signal: {risky_execute.metadata}")
        if risky_execute.metadata.get("normal_followthrough_allowed") is not False or "approval reference for risky recovery step" not in risky_execute.metadata.get("recovery_closure_missing", []):
            raise SystemExit(f"risky checkpoint_recovery_execute should block normal follow-through without approval proof: {risky_execute.metadata}")
        for key in READ_ONLY_FLAGS:
            if risky_execute.metadata.get(key) is not False:
                raise SystemExit(f"risky held checkpoint_recovery_execute unsafe metadata {key}: {risky_execute.metadata}")

        followthrough_result = checkpoint_recovery_followthrough_packet({
            "objective": "continue Jarvis safely",
            "reviewed_step": "recorded reviewed local-safe recovery executor coverage",
            "verification": "continuity smoke passed",
            "receipt_path": str(recovery_receipt_path),
            "receipt_sha256": recovery_receipt_sha256,
            "checkpoint_path": str(recovery_checkpoint_path),
            "checkpoint_sha256": recovery_checkpoint_sha256,
            "stop_condition": "stop if verification drifts",
            "blockers": "none",
        })
        if not followthrough_result.ok or "RECOVERY_FOLLOWTHROUGH_READY" not in followthrough_result.output:
            raise SystemExit(f"checkpoint_recovery_followthrough_packet should be ready: {followthrough_result.output}")
        if followthrough_result.metadata.get("normal_followthrough_allowed") is not True:
            raise SystemExit(f"checkpoint_recovery_followthrough_packet should allow normal follow-through: {followthrough_result.metadata}")
        if followthrough_result.metadata.get("recovery_closure_missing") or followthrough_result.metadata.get("recovery_closure_missing_count") != 0:
            raise SystemExit(f"checkpoint_recovery_followthrough_packet should have complete closure evidence: {followthrough_result.metadata}")
        if followthrough_result.metadata.get("receipt_hash_matches_file") is not True or followthrough_result.metadata.get("checkpoint_hash_matches_file") is not True:
            raise SystemExit(f"checkpoint_recovery_followthrough_packet missed artifact file binding: {followthrough_result.metadata}")
        if followthrough_result.metadata.get("receipt_path_display") != "<local-path>" or followthrough_result.metadata.get("checkpoint_path_display") != "<local-path>":
            raise SystemExit(f"checkpoint_recovery_followthrough_packet missed safe artifact path displays: {followthrough_result.metadata}")
        if str(recovery_receipt_path) in followthrough_result.output or str(recovery_checkpoint_path) in followthrough_result.output:
            raise SystemExit(f"checkpoint_recovery_followthrough_packet leaked raw artifact paths: {followthrough_result.output}")
        if "receipt path: <local-path>" not in followthrough_result.output or "fresh checkpoint: <local-path>" not in followthrough_result.output:
            raise SystemExit(f"checkpoint_recovery_followthrough_packet missed safe artifact displays: {followthrough_result.output}")
        for key in READ_ONLY_FLAGS:
            if followthrough_result.metadata.get(key) is not False:
                raise SystemExit(f"checkpoint_recovery_followthrough_packet unsafe metadata {key}: {followthrough_result.metadata}")

        executed_recovery = checkpoint_recovery_execute({
            "objective": "continue Jarvis safely",
            "reviewed": "true",
            "step": "recorded reviewed local-safe recovery executor coverage",
            "verification": "continuity smoke passed",
            "files": "continuity metadata",
            "outcome": "ready for review",
        })
        if not executed_recovery.ok or "Reviewed local-safe recovery step recorded" not in executed_recovery.output:
            raise SystemExit(f"checkpoint_recovery_execute should record reviewed recovery: {executed_recovery.output}")
        if "Recovery execution contract" not in executed_recovery.output or "verification target" not in executed_recovery.output or "stop condition" not in executed_recovery.output:
            raise SystemExit(f"checkpoint_recovery_execute missed execution contract output: {executed_recovery.output}")
        if "Recovery closure gate" not in executed_recovery.output or "READY_FOR_NORMAL_FOLLOWTHROUGH_REVIEW" not in executed_recovery.output or "normal follow-through allowed: yes" not in executed_recovery.output:
            raise SystemExit(f"checkpoint_recovery_execute missed follow-through closure gate output: {executed_recovery.output}")
        if not executed_recovery.metadata.get("writes_files") or not executed_recovery.metadata.get("writes_notes"):
            raise SystemExit("checkpoint_recovery_execute should mark local continuity writes.")
        if not executed_recovery.metadata.get("receipt_path") or not executed_recovery.metadata.get("checkpoint_path"):
            raise SystemExit(f"checkpoint_recovery_execute should expose receipt/checkpoint paths: {executed_recovery.metadata}")
        if executed_recovery.metadata.get("risky_recovery_without_approval") is not False or executed_recovery.metadata.get("risky_recovery_signal_count") != 0:
            raise SystemExit(f"checkpoint_recovery_execute should record local-safe recovery without risky signals: {executed_recovery.metadata}")
        if not executed_recovery.metadata.get("verification_target") or not executed_recovery.metadata.get("stop_condition"):
            raise SystemExit(f"checkpoint_recovery_execute missed verification target or stop condition metadata: {executed_recovery.metadata}")
        if executed_recovery.metadata.get("recovery_followthrough_gate_state") != "READY_FOR_NORMAL_FOLLOWTHROUGH_REVIEW":
            raise SystemExit(f"checkpoint_recovery_execute missed ready follow-through gate: {executed_recovery.metadata}")
        if executed_recovery.metadata.get("normal_followthrough_allowed") is not True or executed_recovery.metadata.get("recovery_closure_missing"):
            raise SystemExit(f"checkpoint_recovery_execute should allow normal follow-through review after closure proof: {executed_recovery.metadata}")
        if executed_recovery.metadata.get("recovery_closure_required_evidence_count") != len(executed_recovery.metadata.get("recovery_closure_required_evidence", [])):
            raise SystemExit(f"checkpoint_recovery_execute missed closure evidence count: {executed_recovery.metadata}")
        closure_gate = executed_recovery.metadata.get("recovery_followthrough_gate") or {}
        if closure_gate.get("normal_followthrough_allowed") is not True or "approval boundary" not in closure_gate.get("required_evidence", []):
            raise SystemExit(f"checkpoint_recovery_execute missed follow-through gate contract: {closure_gate}")
        contract = executed_recovery.metadata.get("recovery_execution_contract") or {}
        expected_contract_fields = {
            "objective",
            "reviewed",
            "approved_step",
            "verification_target",
            "files",
            "outcome",
            "blockers",
            "risky_recovery_signals",
            "recovery_step_requires_approval",
            "approval_reference_supplied",
            "approval_reference",
            "receipt_path",
            "receipt_sha256",
            "receipt_file_sha256",
            "receipt_hash_matches_file",
            "checkpoint_path",
            "checkpoint_sha256",
            "checkpoint_file_sha256",
            "checkpoint_hash_matches_file",
            "stop_condition",
            "recovery_followthrough_token_sha256",
            "local_safe_recovery_execution_token_sha256",
        }
        if set(contract) != expected_contract_fields:
            raise SystemExit(f"checkpoint_recovery_execute contract fields changed: {contract}")
        if executed_recovery.metadata.get("recovery_execution_contract_field_count") != len(expected_contract_fields):
            raise SystemExit(f"checkpoint_recovery_execute missed contract field count: {executed_recovery.metadata}")
        if not executed_recovery.metadata.get("recovery_executor_recorded"):
            raise SystemExit(f"checkpoint_recovery_execute missed recorded marker: {executed_recovery.metadata}")
        if contract.get("reviewed") is not True or contract.get("approved_step") != executed_recovery.metadata.get("reviewed_step"):
            raise SystemExit(f"checkpoint_recovery_execute contract missed reviewed step: {contract}")
        if contract.get("verification_target") != executed_recovery.metadata.get("verification_target"):
            raise SystemExit(f"checkpoint_recovery_execute contract missed verification target: {contract}")
        if contract.get("receipt_path") != executed_recovery.metadata.get("receipt_path") or contract.get("checkpoint_path") != executed_recovery.metadata.get("checkpoint_path"):
            raise SystemExit(f"checkpoint_recovery_execute contract path aliases diverged: {contract}")
        if contract.get("receipt_file_sha256") != executed_recovery.metadata.get("receipt_file_sha256") or contract.get("receipt_hash_matches_file") is not True:
            raise SystemExit(f"checkpoint_recovery_execute contract missed receipt file binding: {contract}")
        if contract.get("checkpoint_file_sha256") != executed_recovery.metadata.get("checkpoint_file_sha256") or contract.get("checkpoint_hash_matches_file") is not True:
            raise SystemExit(f"checkpoint_recovery_execute contract missed checkpoint file binding: {contract}")
        if contract.get("recovery_followthrough_token_sha256") != executed_recovery.metadata.get("recovery_followthrough_token_sha256"):
            raise SystemExit(f"checkpoint_recovery_execute contract missed recovery token binding: {contract}")
        if contract.get("local_safe_recovery_execution_token_sha256") != executed_recovery.metadata.get("local_safe_recovery_execution_token_sha256"):
            raise SystemExit(f"checkpoint_recovery_execute contract missed local-safe recovery token binding: {contract}")
        if contract.get("recovery_step_requires_approval") is not False or contract.get("approval_reference_supplied") is not False:
            raise SystemExit(f"checkpoint_recovery_execute local-safe contract should not require approval: {contract}")
        if "receipt path:" not in executed_recovery.output or "checkpoint path:" not in executed_recovery.output:
            raise SystemExit(f"checkpoint_recovery_execute did not render contract paths: {executed_recovery.output}")
        if not str(executed_recovery.metadata.get("receipt_path_display") or "").startswith("Reflections/"):
            raise SystemExit(f"checkpoint_recovery_execute missed safe receipt path display: {executed_recovery.metadata}")
        if not str(executed_recovery.metadata.get("checkpoint_path_display") or "").startswith("Reflections/"):
            raise SystemExit(f"checkpoint_recovery_execute missed safe checkpoint path display: {executed_recovery.metadata}")
        if executed_recovery.metadata["receipt_path"] in executed_recovery.output or executed_recovery.metadata["checkpoint_path"] in executed_recovery.output:
            raise SystemExit(f"checkpoint_recovery_execute leaked raw artifact paths: {executed_recovery.output}")
        if executed_recovery.metadata["receipt_path_display"] not in executed_recovery.output or executed_recovery.metadata["checkpoint_path_display"] not in executed_recovery.output:
            raise SystemExit(f"checkpoint_recovery_execute missed safe artifact path displays: {executed_recovery.output}")
        for key in [flag for flag in READ_ONLY_FLAGS if flag not in {"writes_files", "writes_notes"}]:
            if executed_recovery.metadata.get(key) is not False:
                raise SystemExit(f"checkpoint_recovery_execute unsafe metadata {key}: {executed_recovery.metadata}")

        ready_runtime = make_temp_runtime(Path(temp) / "ready-continuation")
        ready_checkpoint = ready_runtime.registry.get("save_work_block_checkpoint").handler({"objective": "ready continuation checkpoint", "limit": 3})
        ready_recovery_checkpoint_path = str(ready_checkpoint.metadata.get("path") or "")
        ready_recovery_checkpoint_sha256 = file_sha256(ready_recovery_checkpoint_path)
        (
            _ready_activity_digest,
            _ready_operator_instruction_supersession_packet,
            _ready_operator_timebox_contract,
            _ready_build_progress_report,
            _ready_build_delta_report,
            _ready_work_block_checkpoint,
            _ready_save_build_progress,
            _ready_save_build_delta,
            _ready_save_work_block_checkpoint,
            _ready_checkpoint_recovery_preview,
            _ready_checkpoint_recovery_apply_packet,
            _ready_checkpoint_recovery_receipt,
            _ready_checkpoint_recovery_execute,
            _ready_checkpoint_recovery_followthrough_packet,
            _ready_checkpoint_recovery_cockpit,
            _ready_autonomy_resume_gate,
            ready_autonomy_continuation_execution_packet,
            ready_autonomy_step_closure_packet,
            _ready_autonomy_cycle_ledger,
            _ready_recent_saved_notes,
        ) = make_continuity_tools(ready_runtime.store, ready_runtime.vault)

        ready_continuation = ready_autonomy_continuation_execution_packet({
            "objective": "continue Jarvis safely",
            "stop_at": "2099-01-01T00:00:00+09:00",
            "current_time": "2026-06-09T03:00:00+09:00",
            "timezone": "Asia/Seoul",
            "reviewed_step": "recorded reviewed local-safe recovery executor coverage",
            "verification": "continuity smoke passed",
            "receipt_path": str(recovery_receipt_path),
            "receipt_sha256": recovery_receipt_sha256,
            "checkpoint_path": ready_recovery_checkpoint_path,
            "checkpoint_sha256": ready_recovery_checkpoint_sha256,
            "stop_condition": "stop if verification drifts",
            "blockers": "none",
            "next_step": "update local continuity metadata",
            "next_verification": "continuity smoke passed",
            "prior_cycle_ledger_token_sha256": prior_cycle_ledger_token_sha256,
        })
        if ready_recovery_checkpoint_path in ready_continuation.output:
            raise SystemExit(f"autonomy continuation packet should not render absolute local checkpoint paths: {ready_continuation.output}")
        if "latest checkpoint: Reflections/" not in ready_continuation.output:
            raise SystemExit(f"autonomy continuation packet missed vault-relative checkpoint display: {ready_continuation.output}")
        if not ready_continuation.ok or "AUTONOMY_CONTINUATION_READY_FOR_ONE_LOCAL_SAFE_STEP" not in ready_continuation.output:
            raise SystemExit(f"autonomy_continuation_execution_packet should be ready: {ready_continuation.output}")
        if ready_continuation.metadata.get("one_local_safe_step_allowed") is not True:
            raise SystemExit(f"autonomy_continuation_execution_packet should allow one local-safe step: {ready_continuation.metadata}")
        if ready_continuation.metadata.get("proposed_next_step_requires_approval") is not False or ready_continuation.metadata.get("missing_blockers"):
            raise SystemExit(f"autonomy_continuation_execution_packet should not require approval for local-safe metadata: {ready_continuation.metadata}")
        if ready_continuation.metadata.get("receipt_hash_matches_file") is not True or ready_continuation.metadata.get("recovery_checkpoint_hash_matches_file") is not True:
            raise SystemExit(f"autonomy_continuation_execution_packet missed recovery artifact file binding: {ready_continuation.metadata}")
        if ready_continuation.metadata.get("post_step_proof_queue_count") != len(ready_continuation.metadata.get("post_step_proof_queue", [])):
            raise SystemExit(f"autonomy_continuation_execution_packet missed post-step proof queue count: {ready_continuation.metadata}")
        if ready_continuation.metadata.get("prior_cycle_ledger_token_present") is not True or ready_continuation.metadata.get("prior_cycle_ledger_token_reusable_for_this_review") is not False:
            raise SystemExit(f"autonomy_continuation_execution_packet missed prior-cycle proof-only boundary: {ready_continuation.metadata}")
        if ready_continuation.metadata.get("prior_cycle_ledger_proof_authorizes_action_now") is not False or ready_continuation.metadata.get("next_step_requires_fresh_cycle_ledger_token") is not True:
            raise SystemExit(f"autonomy_continuation_execution_packet allowed stale cycle proof: {ready_continuation.metadata}")
        for key in READ_ONLY_FLAGS:
            if ready_continuation.metadata.get(key) is not False:
                raise SystemExit(f"autonomy_continuation_execution_packet unsafe metadata {key}: {ready_continuation.metadata}")

        risky_continuation = ready_autonomy_continuation_execution_packet({
            "objective": "continue Jarvis safely",
            "stop_at": "2099-01-01T00:00:00+09:00",
            "current_time": "2026-06-09T03:00:00+09:00",
            "timezone": "Asia/Seoul",
            "reviewed_step": "recorded reviewed local-safe recovery executor coverage",
            "verification": "continuity smoke passed",
            "receipt_path": str(recovery_receipt_path),
            "receipt_sha256": recovery_receipt_sha256,
            "checkpoint_path": ready_recovery_checkpoint_path,
            "checkpoint_sha256": ready_recovery_checkpoint_sha256,
            "stop_condition": "stop if verification drifts",
            "blockers": "none",
            "next_step": "run command python3 --version",
            "next_verification": "verify Python version output",
        })
        if "risky next step needs approval packet" not in risky_continuation.metadata.get("missing_blockers", []):
            raise SystemExit(f"risky autonomy continuation should require approval: {risky_continuation.metadata}")
        if risky_continuation.metadata.get("one_local_safe_step_allowed") is not False or "shell/code" not in risky_continuation.metadata.get("proposed_next_step_risk_signals", []):
            raise SystemExit(f"risky autonomy continuation missed shell/code boundary: {risky_continuation.metadata}")
        for key in READ_ONLY_FLAGS:
            if risky_continuation.metadata.get(key) is not False:
                raise SystemExit(f"risky autonomy_continuation_execution_packet unsafe metadata {key}: {risky_continuation.metadata}")

        held_closure = ready_autonomy_step_closure_packet({
            "objective": "continue Jarvis safely",
            "stop_at": "2099-01-01T00:00:00+09:00",
            "current_time": "2026-06-09T03:00:00+09:00",
            "timezone": "Asia/Seoul",
            "reviewed_step": "recorded reviewed local-safe recovery executor coverage",
            "recovery_verification": "continuity smoke passed",
            "recovery_receipt_path": str(recovery_receipt_path),
            "recovery_receipt_sha256": recovery_receipt_sha256,
            "recovery_checkpoint_path": ready_recovery_checkpoint_path,
            "recovery_checkpoint_sha256": ready_recovery_checkpoint_sha256,
            "stop_condition": "stop if verification drifts",
            "blockers": "none",
            "next_step": "update local continuity metadata",
            "next_verification": "continuity smoke passed",
            "completed_step": "update local continuity metadata",
            "prior_cycle_ledger_token_sha256": prior_cycle_ledger_token_sha256,
        })
        if held_closure.metadata.get("closure_state") != "AUTONOMY_STEP_CLOSURE_HELD":
            raise SystemExit(f"autonomy_step_closure_packet should hold without post-step evidence: {held_closure.metadata}")
        if held_closure.metadata.get("ready_for_next_continuation_review") is not False:
            raise SystemExit(f"held autonomy_step_closure_packet should block next review: {held_closure.metadata}")
        for expected in ["post-step verification evidence", "post-step execution health", "post-step after-action learning"]:
            if expected not in held_closure.metadata.get("missing_blockers", []):
                raise SystemExit(f"held autonomy_step_closure_packet missed blocker {expected!r}: {held_closure.metadata}")
        for key in READ_ONLY_FLAGS:
            if held_closure.metadata.get(key) is not False:
                raise SystemExit(f"held autonomy_step_closure_packet unsafe metadata {key}: {held_closure.metadata}")

        ready_closure = ready_autonomy_step_closure_packet({
            "objective": "continue Jarvis safely",
            "stop_at": "2099-01-01T00:00:00+09:00",
            "current_time": "2026-06-09T03:00:00+09:00",
            "timezone": "Asia/Seoul",
            "reviewed_step": "recorded reviewed local-safe recovery executor coverage",
            "recovery_verification": "continuity smoke passed",
            "recovery_receipt_path": str(recovery_receipt_path),
            "recovery_receipt_sha256": recovery_receipt_sha256,
            "recovery_checkpoint_path": ready_recovery_checkpoint_path,
            "recovery_checkpoint_sha256": ready_recovery_checkpoint_sha256,
            "stop_condition": "stop if verification drifts",
            "blockers": "none",
            "next_step": "update local continuity metadata",
            "next_verification": "continuity smoke passed",
            "completed_step": "update local continuity metadata",
            "post_step_verification": "continuity smoke passed",
            "post_step_receipt_path": str(post_step_receipt_path),
            "post_step_receipt_sha256": post_step_receipt_sha256,
            "post_step_checkpoint_path": str(post_step_checkpoint_path),
            "post_step_checkpoint_sha256": post_step_checkpoint_sha256,
            "execution_health": "execution health reviewed",
            "execution_audit": "execution audit reviewed",
            "after_action_learning": "after-action learning reviewed",
            "prior_cycle_ledger_token_sha256": prior_cycle_ledger_token_sha256,
        })
        if ready_closure.metadata.get("closure_state") != "AUTONOMY_STEP_CLOSURE_READY_FOR_NEXT_CONTINUATION_REVIEW":
            raise SystemExit(f"autonomy_step_closure_packet should be ready: {ready_closure.metadata}")
        if ready_closure.metadata.get("ready_for_next_continuation_review") is not True or ready_closure.metadata.get("missing_blockers"):
            raise SystemExit(f"ready autonomy_step_closure_packet missed ready closure: {ready_closure.metadata}")
        if ready_closure.metadata.get("receipt_hash_matches_file") is not True or ready_closure.metadata.get("recovery_checkpoint_hash_matches_file") is not True:
            raise SystemExit(f"ready autonomy_step_closure_packet missed recovery artifact file binding: {ready_closure.metadata}")
        if ready_closure.metadata.get("post_step_receipt_hash_matches_file") is not True or ready_closure.metadata.get("post_step_checkpoint_hash_matches_file") is not True:
            raise SystemExit(f"ready autonomy_step_closure_packet missed post-step artifact file binding: {ready_closure.metadata}")
        if ready_closure.metadata.get("proof_queue_count") != len(ready_closure.metadata.get("proof_queue", [])):
            raise SystemExit(f"ready autonomy_step_closure_packet missed proof queue count: {ready_closure.metadata}")
        if ready_closure.metadata.get("prior_cycle_ledger_token_present") is not True or ready_closure.metadata.get("prior_cycle_ledger_token_reusable_for_this_closure") is not False:
            raise SystemExit(f"ready autonomy_step_closure_packet missed prior-cycle proof-only closure boundary: {ready_closure.metadata}")
        if ready_closure.metadata.get("prior_cycle_ledger_proof_authorizes_post_step_closure") is not False or ready_closure.metadata.get("next_step_requires_fresh_cycle_ledger_token") is not True:
            raise SystemExit(f"ready autonomy_step_closure_packet allowed stale cycle proof: {ready_closure.metadata}")
        for key in READ_ONLY_FLAGS:
            if ready_closure.metadata.get(key) is not False:
                raise SystemExit(f"ready autonomy_step_closure_packet unsafe metadata {key}: {ready_closure.metadata}")

        ready_cycle_ledger = _ready_autonomy_cycle_ledger({
            "objective": "continue Jarvis safely",
            "stop_at": "2099-01-01T00:00:00+09:00",
            "current_time": "2026-06-09T03:00:00+09:00",
            "timezone": "Asia/Seoul",
            "reviewed_step": "recorded reviewed local-safe recovery executor coverage",
            "recovery_verification": "continuity smoke passed",
            "recovery_receipt_path": str(recovery_receipt_path),
            "recovery_receipt_sha256": recovery_receipt_sha256,
            "recovery_checkpoint_path": ready_recovery_checkpoint_path,
            "recovery_checkpoint_sha256": ready_recovery_checkpoint_sha256,
            "stop_condition": "stop if verification drifts",
            "blockers": "none",
            "next_step": "update local continuity metadata",
            "next_verification": "continuity smoke passed",
            "completed_step": "update local continuity metadata",
            "post_step_verification": "continuity smoke passed",
            "post_step_receipt_path": str(post_step_receipt_path),
            "post_step_receipt_sha256": post_step_receipt_sha256,
            "post_step_checkpoint_path": str(post_step_checkpoint_path),
            "post_step_checkpoint_sha256": post_step_checkpoint_sha256,
            "execution_health": "execution health reviewed",
            "execution_audit": "execution audit reviewed",
            "after_action_learning": "after-action learning reviewed",
            "prior_cycle_ledger_token_sha256": prior_cycle_ledger_token_sha256,
        })
        if ready_cycle_ledger.metadata.get("cycle_state") != "AUTONOMY_CYCLE_LEDGER_READY_FOR_FRESH_REVIEW":
            raise SystemExit(f"autonomy_cycle_ledger should be ready: {ready_cycle_ledger.metadata}")
        if ready_cycle_ledger.metadata.get("prior_cycle_ledger_token_present") is not True or ready_cycle_ledger.metadata.get("prior_cycle_ledger_token_reusable_for_this_cycle") is not False:
            raise SystemExit(f"autonomy_cycle_ledger missed prior-cycle proof-only boundary: {ready_cycle_ledger.metadata}")
        if ready_cycle_ledger.metadata.get("prior_cycle_ledger_proof_authorizes_new_action") is not False or ready_cycle_ledger.metadata.get("next_review_requires_new_cycle_ledger_token") is not True:
            raise SystemExit(f"autonomy_cycle_ledger allowed stale cycle proof: {ready_cycle_ledger.metadata}")
        for key in READ_ONLY_FLAGS:
            if ready_cycle_ledger.metadata.get(key) is not False:
                raise SystemExit(f"autonomy_cycle_ledger unsafe metadata {key}: {ready_cycle_ledger.metadata}")

        notes_result = recent_saved_notes({"limit": 50000})
        if notes_result.metadata.get("limit") != 30:
            raise SystemExit("recent_saved_notes should clamp huge limits.")
        for key in READ_ONLY_FLAGS:
            if notes_result.metadata.get(key) is not False:
                raise SystemExit(f"recent_saved_notes unsafe metadata {key}: {notes_result.metadata}")


if __name__ == "__main__":
    main()
