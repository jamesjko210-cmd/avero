from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from jarvis_v2.scripts.test_runtime import make_temp_runtime
from jarvis_v2.tools import personal as personal_tools
from jarvis_v2.tools.personal import (
    _implementation_review_receipt_contract_ready,
    _metadata_bool,
    _route_lock_token_boundary_ready,
    _route_lock_token_sha256,
    _route_review_contract_ready,
)


def assert_contains(text: str, expected: list[str], label: str) -> None:
    missing = [item for item in expected if item not in text]
    if missing:
        raise SystemExit(f"{label} missing expected text: {missing}")


def assert_read_only_metadata(metadata: dict, label: str) -> None:
    if (
        metadata.get("calls_external_service")
        or metadata.get("reads_personal_data")
        or metadata.get("executes_side_effect")
        or metadata.get("queues_approval")
        or metadata.get("writes_memory")
        or metadata.get("controls_computer")
        or metadata.get("authorizes_execution")
        or metadata.get("authorizes_completion_claim")
        or metadata.get("approval_granted")
    ):
        raise SystemExit(f"{label} must not connect, read data, execute, authorize, queue, write, or control the computer: {metadata}")
    if metadata.get("operator_timeboxes_override_priority") is not True or metadata.get("stop_times_override_priority") is not True:
        raise SystemExit(f"{label} must preserve the operator's stop-time/work-window override contract: {metadata}")


def assert_exact_metadata_bool_helper() -> None:
    cases = [
        (True, False, True),
        (False, True, False),
        ("true", False, False),
        ("false", False, False),
        ("yes", False, False),
        (1, False, False),
        (["ready"], False, False),
        (None, True, True),
        (None, False, False),
    ]
    for value, default, expected in cases:
        observed = _metadata_bool(value, default=default)
        if observed is not expected:
            raise SystemExit(f"_metadata_bool accepted non-exact readiness value {value!r}: expected {expected}, got {observed}")


def contains_raw_local_path(value: object) -> bool:
    if isinstance(value, str):
        return any(fragment in value for fragment in ["/\x55sers/", "/private/", "/var/folders/", "/tmp/"])
    if isinstance(value, dict):
        return any(contains_raw_local_path(item) for item in value.values())
    if isinstance(value, (list, tuple, set)):
        return any(contains_raw_local_path(item) for item in value)
    return False


def assert_local_paths_redacted(result, label: str) -> None:
    if contains_raw_local_path(result.output):
        raise SystemExit(f"{label} leaked raw local path output: {result.output}")
    metadata = result.metadata or {}
    if contains_raw_local_path(metadata):
        raise SystemExit(f"{label} leaked raw local path metadata: {metadata}")
    if "<local-path>" not in result.output and not contains_redacted_local_path(metadata):
        raise SystemExit(f"{label} should include a redacted local-path marker: output={result.output} metadata={metadata}")


def assert_reminder_outcome_unknown(result, label: str) -> None:
    if result.ok:
        raise SystemExit(f"{label} should fail closed")
    for expected in [
        "may have completed",
        "Check Reminders for the exact title",
        "do not automatically retry",
        "setup check",
        "Reminders Automation access",
        "new approved request",
    ]:
        if expected not in result.output:
            raise SystemExit(
                f"{label} missed outcome-unknown recovery guidance {expected}: "
                f"{result.output}"
            )
    metadata = result.metadata or {}
    expected_truth = {
        "outcome_known": False,
        "outcome_unknown": True,
        "execution_outcome_unknown": True,
        "side_effect_possible": True,
        "retry_safe": False,
        "automatic_retry_allowed": False,
        "authorizes_retry": False,
    }
    for key, expected in expected_truth.items():
        if metadata.get(key) is not expected:
            raise SystemExit(
                f"{label} changed {key}: expected {expected!r}, got {metadata.get(key)!r}"
            )
    if (
        metadata.get("next_command") != "setup check"
        or metadata.get("recovery_commands") != ["setup check"]
        or metadata.get("recovery_guidance")
        != {
            "version": 1,
            "action": result.output,
            "commands": ["setup check"],
        }
    ):
        raise SystemExit(f"{label} lost bounded machine-readable recovery: {metadata}")


def contains_redacted_local_path(value: object) -> bool:
    if isinstance(value, str):
        return "<local-path>" in value
    if isinstance(value, dict):
        return any(contains_redacted_local_path(item) for item in value.values())
    if isinstance(value, (list, tuple, set)):
        return any(contains_redacted_local_path(item) for item in value)
    return False


def main() -> None:
    assert_exact_metadata_bool_helper()
    with TemporaryDirectory(prefix="jarvis-personal-") as temp:
        runtime = make_temp_runtime(Path(temp))
        cases = [
            "integration status",
            "personal integrations",
            "what personal integrations are active",
            "integration readiness",
            "integration readiness: calendar",
            "integration execution matrix",
            "integration execution matrix: email",
            "legacy connector migration audit",
            "old Jarvis migration audit: browser calendar email",
            "integration adapter manifest",
            "integration adapter manifest: email",
            "integration adapter probe: email -> search mailbox metadata; target inbox; time last 7 days; data metadata-only; verification fake rows only",
            "integration adapter probe: email -> search mailbox metadata; target inbox; time last 7 days; data full body; verification fake rows only",
            "integration adapter probe: email -> send draft reply; target thread 123; time today; data metadata-only; verification fake rows only",
            "integration adapter acceptance: email",
            "integration contract: calendar",
            "integration boundary contract: email",
            "connector contract: messages",
            "integration action preview: email -> search mailbox metadata",
            "integration action preview: email -> send draft reply",
            "connector action preview: browser -> summarize logged-in page",
            "integration scope packet: email -> search mailbox metadata; target inbox; time last 7 days; data metadata-only",
            "integration scope packet: calendar -> create event; target the operator calendar; time tomorrow 10am; data event title and attendee",
            "integration scope packet: browser -> summarize logged-in page",
            "integration dry run contract: email -> send draft reply; target thread 123; time today; data draft-only; verification confirm not sent",
            "integration dry run contract: email -> search mailbox metadata; target inbox; time last 7 days; data metadata-only; verification fake rows only",
            "integration dry run contract: browser -> summarize logged-in page",
            "integration runbook: email -> send draft reply; target thread 123; time today; data draft-only; verification confirm not sent; rollback discard draft",
            "integration runbook: email -> search mailbox metadata; target inbox; time last 7 days; data metadata-only; verification fake rows only; rollback no connector call",
            "integration runbook: browser -> summarize logged-in page",
            "integration promotion gate: email -> send draft reply; target thread 123; time today; data draft-only; verification confirm not sent; rollback discard draft; tests blocked send smoke; audit tool run receipt",
            "integration promotion gate: email -> search mailbox metadata; target inbox; time last 7 days; data metadata-only; verification fake rows only; rollback no connector call; tests adapter acceptance passed; audit adapter acceptance receipt",
            "integration promotion gate: browser -> summarize logged-in page",
            "integration implementation spec: email -> send draft reply; target thread 123; time today; data draft-only; verification confirm not sent; rollback discard draft; tests blocked send smoke; audit tool run receipt",
            "integration implementation spec: email -> search mailbox metadata; target inbox; time last 7 days; data metadata-only; verification fake rows only; rollback no connector call; tests adapter acceptance passed; audit adapter acceptance receipt",
            "integration implementation spec: browser -> summarize logged-in page",
            "integration preflight contract: email -> send draft reply; target thread 123; time today; data draft-only; verification confirm not sent; rollback discard draft; tests blocked send smoke; audit tool run receipt",
            "integration preflight contract: email -> search mailbox metadata; target inbox; time last 7 days; data metadata-only; verification fake rows only; rollback no connector call; tests adapter acceptance passed; audit adapter acceptance receipt",
            "integration preflight contract: browser -> summarize logged-in page",
            "integration enablement gate: email -> send draft reply; target thread 123; time today; data draft-only; verification confirm not sent; rollback discard draft; tests blocked send smoke; audit tool run receipt; acceptance gate passed with receipts",
            "integration enablement gate: email -> search mailbox metadata; target inbox; time last 7 days; data metadata-only; verification fake rows only; rollback no connector call; tests adapter acceptance passed; audit adapter acceptance receipt; acceptance gate passed",
            "integration enablement gate: browser -> summarize logged-in page",
            "integration rehearsal receipt: email -> send draft reply; target thread 123; time today; data draft-only; verification confirm not sent; rollback discard draft; tests blocked send smoke; audit tool run receipt; acceptance gate passed with receipts",
            "integration rehearsal receipt: email -> search mailbox metadata; target inbox; time last 7 days; data metadata-only; verification fake rows only; rollback no connector call; tests adapter acceptance passed; audit adapter acceptance receipt; acceptance gate passed",
            "integration rehearsal receipt: browser -> summarize logged-in page",
            "integration metadata preview: email -> search mailbox metadata; target inbox; time last 7 days; data metadata-only; verification fake rows only; tests blocked full body smoke; audit metadata preview receipt; acceptance gate passed",
            "integration metadata preview: email -> search mailbox metadata; target inbox; time last 7 days; data full body; verification fake rows only; tests blocked full body smoke; audit metadata preview receipt; acceptance gate passed",
            "integration metadata preview: email -> send draft reply; target thread 123; time today; data metadata-only; verification fake rows only; tests blocked send smoke; audit metadata preview receipt; acceptance gate passed",
            "integration proof bundle: email -> search mailbox metadata; target inbox; time last 7 days; data metadata-only; verification fake rows only; rollback no connector call; tests adapter acceptance passed; audit adapter acceptance receipt; acceptance gate passed",
            "integration proof bundle: browser -> summarize logged-in page",
            "integration implementation review: email -> search mailbox metadata; target inbox; time last 7 days; data metadata-only; verification fake rows only; rollback no connector call; tests adapter acceptance passed; audit adapter acceptance receipt; acceptance gate passed",
            "integration implementation review: email -> search mailbox metadata; target inbox; time last 7 days; data metadata-only; verification fake rows only; rollback no connector call; tests adapter acceptance passed; audit adapter acceptance receipt; acceptance gate passed; status status api smoke passed",
            "integration implementation review: browser -> summarize logged-in page",
            "integration route lock: email -> search mailbox metadata; target inbox; time last 7 days; data metadata-only; verification fake rows only; rollback no connector call; tests adapter acceptance passed; audit adapter acceptance receipt; acceptance gate passed",
            "integration route lock: email -> search mailbox metadata; target inbox; time last 7 days; data metadata-only; verification fake rows only; rollback no connector call; tests adapter acceptance passed; audit adapter acceptance receipt; acceptance gate passed; status status api smoke passed",
            "integration route lock: email -> search mailbox metadata; target inbox; time last 7 days; data metadata-only; verification fake rows only; rollback no connector call; tests adapter acceptance passed; audit adapter acceptance receipt; acceptance gate passed; status status api smoke passed; expected scope hash deadbeefdeadbeef",
            "integration route lock: email -> search mailbox metadata; target inbox; time last 7 days; data metadata-only; verification fake rows only; rollback no connector call; tests adapter acceptance passed; audit adapter acceptance receipt; acceptance gate passed; status status api smoke passed; expected scope hash not-a-scope-hash",
            "integration route lock: browser -> summarize logged-in page",
            "integration migration plan: calendar",
            "integration migration plan: email",
            "connector migration plan: messages",
        ]
        for case in cases:
            result = runtime.handle(case)
            status = "ok" if result.verified else "blocked"
            print(f"[{status}] {case}")
            print(result.response[:2200])
            print()
            if not result.verified:
                raise SystemExit(f"Expected '{case}' to run as read-only.")
            if "readiness" in case:
                assert_contains(
                    result.response,
                    [
                        "Jarvis personal integration readiness report",
                        "This is read-only",
                        "Readiness rule",
                        "Connector readiness",
                        "safest next command",
                        "Suggested migration order",
                        "Hard stops",
                    ],
                    case,
                )
                metadata = result.tool_results[0].metadata
                assert_read_only_metadata(metadata, "Integration readiness")
                handoff = metadata.get("readiness_report_handoff")
                if not isinstance(handoff, dict):
                    raise SystemExit(f"Expected structured readiness report handoff: {metadata}")
                handoff_requirements = handoff.get("proof_requirements") or {}
                handoff_rule = handoff.get("routing_rule") or {}
                handoff_boundaries = handoff.get("boundaries") or {}
                if (
                    metadata.get("readiness_report_handoff_ready") is not True
                    or handoff.get("readiness_report_handoff_ready") is not True
                    or metadata.get("readiness_report_handoff_ready") != handoff.get("handoff_ready")
                    or metadata.get("readiness_report_ready_for_operator") is not True
                    or handoff.get("ready_for_operator") is not True
                    or metadata.get("readiness_report_state_changed") is not False
                    or handoff.get("state_changed") is not False
                    or metadata.get("readiness_report_changed") != []
                    or handoff.get("changed") != []
                    or metadata.get("readiness_report_content_in_handoff") is not True
                    or handoff.get("content_in_handoff") is not True
                    or metadata.get("readiness_report_authorizes_execution") is not False
                    or metadata.get("readiness_report_authorizes_completion_claim") is not False
                    or metadata.get("readiness_report_approval_granted") is not False
                    or handoff.get("authorizes_execution") is not False
                    or handoff.get("authorizes_completion_claim") is not False
                    or handoff.get("approval_granted") is not False
                    or metadata.get("readiness_report_boundaries") != handoff_boundaries
                    or handoff.get("connectors_reviewed") != metadata.get("connectors_reviewed")
                    or handoff.get("readiness_rows") != metadata.get("readiness_rows")
                    or handoff.get("recommendations") != metadata.get("recommendations")
                    or len(handoff.get("readiness_rows") or []) != metadata.get("connectors_reviewed")
                    or not handoff.get("migration_order")
                    or handoff_requirements.get("requires_boundary_contract") is not True
                    or handoff_requirements.get("requires_action_preview") is not True
                    or handoff_requirements.get("requires_audit_logging") is not True
                    or handoff_requirements.get("requires_blocked_action_smoke_tests") is not True
                    or handoff_requirements.get("requires_fresh_route_review") is not True
                    or handoff_rule.get("start_with_narrow_read_only_or_draft_only_tools") is not True
                    or handoff_rule.get("personal_data_reads_approval_gated") is not True
                    or handoff_rule.get("side_effects_approval_gated") is not True
                    or handoff_rule.get("natural_language_routing_enabled")
                    or handoff_boundaries.get("natural_language_routing_enabled")
                    or handoff_boundaries.get("calls_external_service")
                    or handoff_boundaries.get("reads_personal_data")
                    or handoff_boundaries.get("executes_side_effect")
                    or handoff_boundaries.get("authorizes_account_access")
                    or handoff_boundaries.get("authorizes_route_unlock")
                    or handoff_boundaries.get("authorizes_natural_language_routing")
                    or handoff_boundaries.get("authorizes_personal_data_read")
                    or handoff_boundaries.get("authorizes_side_effect")
                    or handoff_boundaries.get("authorizes_approval")
                    or handoff_boundaries.get("authorizes_model_call")
                    or handoff_boundaries.get("authorizes_tool_execution")
                    or handoff_boundaries.get("authorizes_external_service")
                    or handoff_boundaries.get("reusable_for_other_scope")
                    or handoff.get("next_step_requires_boundary_contract") is not True
                    or handoff.get("next_step_requires_action_preview") is not True
                    or handoff.get("next_route_lock_requires_fresh_review") is not True
                ):
                    raise SystemExit(f"Readiness report handoff diverged from flat metadata or safety contract: {metadata}")
                if case == "integration readiness" and metadata.get("connectors_reviewed", 0) < 6:
                    raise SystemExit("Integration readiness should review all known connectors by default.")
                if case == "integration readiness: calendar" and metadata.get("connectors_reviewed") != 1:
                    raise SystemExit("Integration readiness should support one-connector focus.")
            elif "legacy connector" in case.lower() or "old jarvis migration audit" in case.lower():
                assert_contains(
                    result.response,
                    [
                        "Jarvis legacy connector migration audit",
                        "This is read-only",
                        "Claude handoff",
                        "TASK-F contact resolver wiring is marked done",
                        "TASK-G morning brief Telegram scheduling is marked done",
                        "Companion/team-cockpit architecture was rejected",
                        "Legacy surfaces reviewed",
                        "browser",
                        "calendar",
                        "email",
                        "Required proof before any legacy behavior is enabled",
                        "Boundary contract",
                        "disabled adapter acceptance",
                        "route lock",
                        "Hard stops",
                    ],
                    case,
                )
                metadata = result.tool_results[0].metadata
                handoff = metadata.get("legacy_connector_migration_audit_handoff")
                if not isinstance(handoff, dict):
                    raise SystemExit(f"Expected structured legacy migration audit handoff: {metadata}")
                handoff_state = str(handoff.get("claude_handoff_state") or "")
                if "TASK-G morning brief Telegram scheduling done" not in handoff_state:
                    raise SystemExit(f"Legacy migration audit handoff should mark TASK-G done: {metadata}")
                assert_read_only_metadata(metadata, "Legacy connector migration audit")
                handoff = metadata.get("legacy_connector_migration_audit_handoff")
                if not isinstance(handoff, dict):
                    raise SystemExit(f"Expected structured legacy connector migration audit handoff: {metadata}")
                handoff_rows = handoff.get("migration_rows") or []
                handoff_requirements = handoff.get("proof_requirements") or {}
                handoff_boundaries = handoff.get("boundaries") or {}
                if (
                    metadata.get("legacy_connector_migration_audit_handoff_ready") is not True
                    or handoff.get("legacy_connector_migration_audit_handoff_ready") is not True
                    or handoff.get("handoff_ready") is not True
                    or metadata.get("legacy_connector_migration_audit_ready_for_operator") is not True
                    or handoff.get("ready_for_operator") is not True
                    or metadata.get("legacy_connector_migration_audit_state_changed") is not False
                    or handoff.get("state_changed") is not False
                    or metadata.get("legacy_connector_migration_audit_changed") != []
                    or handoff.get("changed") != []
                    or metadata.get("legacy_connector_migration_audit_content_in_handoff") is not False
                    or handoff.get("content_in_handoff") is not False
                    or handoff.get("authorizes_execution") is not False
                    or handoff.get("authorizes_completion_claim") is not False
                    or handoff.get("approval_granted") is not False
                    or metadata.get("legacy_surfaces_reviewed") != 3
                    or handoff.get("legacy_surfaces_reviewed") != metadata.get("legacy_surfaces_reviewed")
                    or handoff.get("connectors") != metadata.get("connectors_reviewed")
                    or [row.get("connector") for row in handoff_rows] != ["browser", "calendar", "email"]
                    or handoff_requirements.get("requires_boundary_contract") is not True
                    or handoff_requirements.get("requires_disabled_adapter_acceptance") is not True
                    or handoff_requirements.get("requires_route_lock") is not True
                    or handoff_requirements.get("requires_focused_smoke") is not True
                    or handoff_requirements.get("requires_aggregate_smoke") is not True
                    or handoff_requirements.get("requires_fresh_approval_for_personal_data_or_side_effects") is not True
                    or metadata.get("natural_language_routing_enabled")
                    or metadata.get("route_locks_enabled")
                    or not all("required_commands" in row for row in handoff_rows)
                    or not all("route lock" in " ".join(row.get("required_commands") or []) for row in handoff_rows)
                    or handoff_boundaries.get("natural_language_routing_enabled")
                    or handoff_boundaries.get("calls_external_service")
                    or handoff_boundaries.get("reads_personal_data")
                    or handoff_boundaries.get("executes_side_effect")
                    or handoff_boundaries.get("controls_computer")
                    or handoff_boundaries.get("queues_approval")
                    or handoff_boundaries.get("authorizes_account_access")
                    or handoff_boundaries.get("authorizes_route_unlock")
                    or handoff_boundaries.get("authorizes_natural_language_routing")
                    or handoff_boundaries.get("authorizes_personal_data_read")
                    or handoff_boundaries.get("authorizes_side_effect")
                    or handoff_boundaries.get("authorizes_approval")
                    or handoff_boundaries.get("authorizes_tool_execution")
                    or handoff_boundaries.get("reusable_for_other_scope")
                    or metadata.get("legacy_connector_migration_audit_boundaries") != handoff_boundaries
                    or metadata.get("legacy_connector_migration_audit_next_safe_commands") != handoff.get("next_safe_commands")
                ):
                    raise SystemExit(f"Legacy connector migration audit handoff diverged from safety contract: {metadata}")
            elif "execution matrix" in case:
                assert_contains(
                    result.response,
                    [
                        "Jarvis personal integration execution matrix",
                        "This is read-only",
                        "Global connector rule",
                        "Connector lanes",
                        "disabled",
                        "Proof required before enabling any connector",
                        "execution health report",
                        "Hard stops",
                    ],
                    case,
                )
                metadata = result.tool_results[0].metadata
                assert_read_only_metadata(metadata, "Integration execution matrix")
                handoff = metadata.get("execution_matrix_handoff")
                if not isinstance(handoff, dict):
                    raise SystemExit(f"Expected structured execution matrix handoff: {metadata}")
                handoff_rule = handoff.get("global_connector_rule") or {}
                handoff_requirements = handoff.get("proof_requirements") or {}
                handoff_boundaries = handoff.get("boundaries") or {}
                if (
                    metadata.get("execution_matrix_handoff_ready") is not True
                    or handoff.get("execution_matrix_handoff_ready") is not True
                    or metadata.get("execution_matrix_handoff_ready") != handoff.get("handoff_ready")
                    or metadata.get("execution_matrix_ready_for_operator") is not True
                    or handoff.get("ready_for_operator") is not True
                    or metadata.get("execution_matrix_state_changed") is not False
                    or handoff.get("state_changed") is not False
                    or metadata.get("execution_matrix_changed") != []
                    or handoff.get("changed") != []
                    or metadata.get("execution_matrix_content_in_handoff") is not True
                    or handoff.get("content_in_handoff") is not True
                    or metadata.get("execution_matrix_authorizes_execution") is not False
                    or metadata.get("execution_matrix_authorizes_completion_claim") is not False
                    or metadata.get("execution_matrix_approval_granted") is not False
                    or handoff.get("authorizes_execution") is not False
                    or handoff.get("authorizes_completion_claim") is not False
                    or handoff.get("approval_granted") is not False
                    or metadata.get("execution_matrix_boundaries") != handoff_boundaries
                    or metadata.get("execution_matrix_next_safe_commands") != handoff.get("next_safe_commands")
                    or not metadata.get("execution_matrix_next_safe_commands")
                    or handoff.get("connector") != metadata.get("connector")
                    or handoff.get("connectors_reviewed") != metadata.get("connectors_reviewed")
                    or handoff.get("lanes") != metadata.get("lanes")
                    or handoff.get("adapter_states") != metadata.get("adapter_states")
                    or handoff.get("safest_commands") != metadata.get("safest_commands")
                    or not set(metadata.get("safest_commands") or []).issubset(set(metadata.get("execution_matrix_next_safe_commands") or []))
                    or len(handoff.get("lane_rows") or []) != metadata.get("connectors_reviewed")
                    or handoff_requirements.get("requires_preflight_contract") is not True
                    or handoff_requirements.get("requires_enablement_gate") is not True
                    or handoff_requirements.get("requires_rehearsal_receipt") is not True
                    or handoff_requirements.get("requires_metadata_preview_for_metadata_paths") is not True
                    or handoff_requirements.get("requires_smoke_tests") != metadata.get("requires_smoke_tests")
                    or handoff_requirements.get("requires_acceptance_gate") != metadata.get("requires_acceptance_gate")
                    or handoff_requirements.get("requires_execution_health_report") != metadata.get("requires_execution_health_report")
                    or handoff_requirements.get("requires_fresh_route_review") is not True
                    or handoff_rule.get("metadata_or_draft_rehearsals_only") is not True
                    or handoff_rule.get("personal_data_and_side_effects_require_exact_scope") is not True
                    or handoff_rule.get("one_shot_approval_required_for_personal_data_or_side_effects") is not True
                    or handoff_rule.get("approval_chain_proof_required") is not True
                    or handoff_rule.get("audit_receipt_required") is not True
                    or handoff_rule.get("verification_required") is not True
                    or handoff_rule.get("rollback_or_stop_evidence_required") is not True
                    or handoff_rule.get("natural_language_auto_routing_disabled_until_smokes_pass") is not True
                    or "missing scope" not in (handoff.get("blocked_test_requirements") or [])
                    or "side effect without approval" not in (handoff.get("blocked_test_requirements") or [])
                    or handoff_boundaries.get("natural_language_routing_enabled")
                    or handoff_boundaries.get("calls_external_service")
                    or handoff_boundaries.get("reads_personal_data")
                    or handoff_boundaries.get("executes_side_effect")
                    or handoff_boundaries.get("authorizes_account_access")
                    or handoff_boundaries.get("authorizes_route_unlock")
                    or handoff_boundaries.get("authorizes_natural_language_routing")
                    or handoff_boundaries.get("authorizes_personal_data_read")
                    or handoff_boundaries.get("authorizes_side_effect")
                    or handoff_boundaries.get("authorizes_approval")
                    or handoff_boundaries.get("authorizes_model_call")
                    or handoff_boundaries.get("authorizes_tool_execution")
                    or handoff_boundaries.get("authorizes_external_service")
                    or handoff_boundaries.get("authorizes_execution")
                    or handoff_boundaries.get("authorizes_completion_claim")
                    or handoff_boundaries.get("approval_granted")
                    or handoff_boundaries.get("reusable_for_other_scope")
                    or handoff.get("next_step_requires_preflight_contract") is not True
                    or handoff.get("next_step_requires_enablement_gate") is not True
                    or handoff.get("next_route_lock_requires_fresh_review") is not True
                ):
                    raise SystemExit(f"Execution matrix handoff diverged from flat metadata or safety contract: {metadata}")
                if metadata.get("natural_language_routing_enabled"):
                    raise SystemExit(f"Integration execution matrix should keep NL routing disabled: {metadata}")
                if not metadata.get("requires_smoke_tests") or not metadata.get("requires_acceptance_gate") or not metadata.get("requires_execution_health_report"):
                    raise SystemExit(f"Integration execution matrix missed proof metadata: {metadata}")
                if case == "integration execution matrix" and metadata.get("connectors_reviewed", 0) < 6:
                    raise SystemExit("Integration execution matrix should review all known connectors by default.")
                if case == "integration execution matrix: email" and metadata.get("connectors_reviewed") != 1:
                    raise SystemExit("Integration execution matrix should support one-connector focus.")
            elif "adapter manifest" in case:
                assert_contains(
                    result.response,
                    [
                        "Jarvis personal integration adapter manifest",
                        "This is read-only",
                        "Adapter rule",
                        "Adapter stubs",
                        "disabled",
                        "blocked side-effect path",
                        "row schema: id, timestamp, label, source",
                        "row limit: 2",
                        "bounded rows: yes",
                        "blocked payload fields:",
                        "Implementation order",
                        "Hard stops",
                    ],
                    case,
                )
                metadata = result.tool_results[0].metadata
                assert_read_only_metadata(metadata, "Integration adapter manifest")
                handoff = metadata.get("adapter_manifest_handoff")
                if not isinstance(handoff, dict):
                    raise SystemExit(f"Expected structured adapter manifest handoff: {metadata}")
                handoff_contract = handoff.get("row_contract") or {}
                handoff_requirements = handoff.get("proof_requirements") or {}
                handoff_boundaries = handoff.get("boundaries") or {}
                if (
                    metadata.get("adapter_manifest_handoff_ready") is not True
                    or handoff.get("adapter_manifest_handoff_ready") is not True
                    or metadata.get("adapter_manifest_handoff_ready") != handoff.get("handoff_ready")
                    or handoff.get("handoff_ready") is not True
                    or metadata.get("adapter_manifest_ready_for_operator") != handoff.get("ready_for_operator")
                    or metadata.get("adapter_manifest_ready_for_operator") is not True
                    or handoff.get("ready_for_operator") is not True
                    or metadata.get("adapter_manifest_state_changed") is not False
                    or metadata.get("adapter_manifest_changed") != []
                    or metadata.get("adapter_manifest_content_in_handoff") is not True
                    or metadata.get("adapter_manifest_authorizes_execution")
                    or metadata.get("adapter_manifest_authorizes_completion_claim")
                    or metadata.get("adapter_manifest_approval_granted")
                    or metadata.get("adapter_manifest_boundaries") != handoff_boundaries
                    or metadata.get("adapter_manifest_next_safe_commands") != handoff.get("next_safe_commands")
                    or not metadata.get("adapter_manifest_next_safe_commands")
                    or handoff.get("state_changed") is not False
                    or handoff.get("changed") != []
                    or handoff.get("content_in_handoff") is not True
                    or handoff.get("authorizes_execution")
                    or handoff.get("authorizes_completion_claim")
                    or handoff.get("approval_granted")
                    or handoff.get("connector") != metadata.get("connector")
                    or handoff.get("connectors_reviewed") != metadata.get("connectors_reviewed")
                    or handoff.get("manifest_rows") != metadata.get("manifests")
                    or handoff.get("adapter_states") != metadata.get("adapter_states")
                    or handoff.get("future_tools") != metadata.get("future_tools")
                    or handoff.get("fixture_names") != metadata.get("fixture_names")
                    or handoff_contract.get("sample_row_schema") != metadata.get("sample_row_schema")
                    or handoff_contract.get("sample_row_limit") != metadata.get("sample_row_limit")
                    or handoff_contract.get("blocked_payload_fields") != metadata.get("blocked_payload_fields")
                    or handoff_contract.get("all_manifest_row_contracts_present") != metadata.get("all_manifest_row_contracts_present")
                    or handoff_requirements.get("requires_metadata_preview") != metadata.get("requires_metadata_preview")
                    or handoff_requirements.get("requires_blocked_full_content_test") != metadata.get("requires_blocked_full_content_test")
                    or handoff_requirements.get("requires_blocked_side_effect_test") != metadata.get("requires_blocked_side_effect_test")
                    or handoff_requirements.get("requires_acceptance_gate") != metadata.get("requires_acceptance_gate")
                    or handoff_requirements.get("requires_execution_health_report") != metadata.get("requires_execution_health_report")
                    or handoff_requirements.get("requires_fresh_route_review") is not True
                    or not handoff.get("implementation_order")
                    or handoff_boundaries.get("natural_language_routing_enabled")
                    or handoff_boundaries.get("adapter_default_state") != "disabled"
                    or handoff_boundaries.get("calls_external_service")
                    or handoff_boundaries.get("reads_personal_data")
                    or handoff_boundaries.get("executes_side_effect")
                    or handoff_boundaries.get("writes_memory")
                    or handoff_boundaries.get("authorizes_account_access")
                    or handoff_boundaries.get("authorizes_route_unlock")
                    or handoff_boundaries.get("authorizes_natural_language_routing")
                    or handoff_boundaries.get("authorizes_personal_data_read")
                    or handoff_boundaries.get("authorizes_side_effect")
                    or handoff_boundaries.get("authorizes_approval")
                    or handoff_boundaries.get("authorizes_execution")
                    or handoff_boundaries.get("authorizes_completion_claim")
                    or handoff_boundaries.get("approval_granted")
                    or handoff_boundaries.get("authorizes_model_call")
                    or handoff_boundaries.get("authorizes_tool_execution")
                    or handoff_boundaries.get("authorizes_external_service")
                    or handoff_boundaries.get("reusable_for_other_scope")
                    or handoff.get("next_step_requires_adapter_probe") is not True
                    or handoff.get("next_step_requires_adapter_acceptance") is not True
                    or handoff.get("next_route_lock_requires_fresh_review") is not True
                ):
                    raise SystemExit(f"Adapter manifest handoff diverged from flat metadata or safety contract: {metadata}")
                if metadata.get("natural_language_routing_enabled"):
                    raise SystemExit(f"Integration adapter manifest should keep NL routing disabled: {metadata}")
                for key in ["requires_metadata_preview", "requires_blocked_full_content_test", "requires_blocked_side_effect_test", "requires_acceptance_gate", "requires_execution_health_report"]:
                    if not metadata.get(key):
                        raise SystemExit(f"Integration adapter manifest missed proof metadata {key}: {metadata}")
                if (
                    metadata.get("sample_row_schema") != ["id", "timestamp", "label", "source"]
                    or metadata.get("sample_row_limit") != 2
                    or metadata.get("all_manifest_row_contracts_present") is not True
                ):
                    raise SystemExit(f"Integration adapter manifest missed row contract metadata: {metadata}")
                if case == "integration adapter manifest" and metadata.get("connectors_reviewed", 0) < 6:
                    raise SystemExit("Integration adapter manifest should review all known connectors by default.")
                if case == "integration adapter manifest: email" and metadata.get("connectors_reviewed") != 1:
                    raise SystemExit("Integration adapter manifest should support one-connector focus.")
                blocked_manifest = personal_tools._adapter_manifest_handoff_payload(
                    requested_connector="email",
                    rows=[
                        {
                            "connector": "email",
                            "adapter_name": "email_metadata_adapter",
                            "future_tool": "future_email_metadata_preview",
                            "fixture_name": "email_metadata_fixture",
                            "safe_action": "search mailbox metadata",
                            "blocked_action": "send draft reply",
                            "adapter_state": "disabled",
                            "risk": "READ_ONLY_METADATA_ONLY",
                            "enablement_gate": "integration_enablement_gate",
                            "sample_row_schema": ["id", "timestamp"],
                            "sample_row_limit": 2,
                            "sample_rows_bounded": False,
                            "blocked_payload_fields": ["body"],
                        }
                    ],
                    sample_row_schema=["id", "timestamp", "label", "source"],
                    sample_row_limit=2,
                    blocked_payload_fields=["body"],
                    all_manifest_row_contracts_present=False,
                )
                blocked_boundaries = blocked_manifest.get("boundaries") or {}
                if (
                    blocked_manifest.get("handoff_ready") is not True
                    or blocked_manifest.get("ready_for_operator") is not True
                    or (blocked_manifest.get("row_contract") or {}).get("all_manifest_row_contracts_present") is not False
                    or blocked_manifest.get("authorizes_execution")
                    or blocked_manifest.get("authorizes_completion_claim")
                    or blocked_manifest.get("approval_granted")
                    or blocked_boundaries.get("natural_language_routing_enabled")
                    or blocked_boundaries.get("adapter_default_state") != "disabled"
                    or blocked_boundaries.get("calls_external_service")
                    or blocked_boundaries.get("reads_personal_data")
                    or blocked_boundaries.get("executes_side_effect")
                    or blocked_boundaries.get("writes_memory")
                    or blocked_boundaries.get("authorizes_account_access")
                    or blocked_boundaries.get("authorizes_route_unlock")
                    or blocked_boundaries.get("authorizes_natural_language_routing")
                    or blocked_boundaries.get("authorizes_personal_data_read")
                    or blocked_boundaries.get("authorizes_side_effect")
                    or blocked_boundaries.get("authorizes_approval")
                    or blocked_boundaries.get("authorizes_execution")
                    or blocked_boundaries.get("authorizes_completion_claim")
                    or blocked_boundaries.get("approval_granted")
                    or blocked_boundaries.get("authorizes_model_call")
                    or blocked_boundaries.get("authorizes_tool_execution")
                    or blocked_boundaries.get("authorizes_external_service")
                    or blocked_boundaries.get("reusable_for_other_scope")
                    or not blocked_manifest.get("next_safe_commands")
                ):
                    raise SystemExit(f"Blocked adapter manifest handoff should remain consumable and non-authorizing: {blocked_manifest}")
            elif case.startswith("integration adapter acceptance"):
                assert_contains(
                    result.response,
                    [
                        "Jarvis personal integration adapter acceptance report",
                        "This is read-only",
                        "Acceptance state: ADAPTER_ACCEPTANCE_PASSED",
                        "Cases passed: 4/4",
                        "Acceptance receipt",
                        "acceptance receipt sha256:",
                        "receipt boundary: proof-only, enablement-review-required, account-access-not-authorized, route-unlock-not-authorized, fresh-scope-review-required",
                        "receipt rows: 5",
                        "metadata happy path: pass",
                        "full content blocked: pass",
                        "side effect blocked: pass",
                        "missing scope blocked: pass",
                        "row schema: id, timestamp, label, source",
                        "row limit: 2",
                        "Natural-language routing remains disabled",
                        "Next required commands",
                        "Hard stops",
                    ],
                    case,
                )
                if "Next proof commands" in result.response:
                    raise SystemExit(f"{case} should render adapter handoff as next required commands, not next proof commands: {result.response}")
                metadata = result.tool_results[0].metadata
                assert_read_only_metadata(metadata, "Integration adapter acceptance")
                handoff = metadata.get("adapter_acceptance_handoff")
                if not isinstance(handoff, dict):
                    raise SystemExit(f"Expected structured adapter acceptance handoff: {metadata}")
                receipt_sha256 = metadata.get("adapter_acceptance_receipt_sha256")
                receipt_rows = metadata.get("adapter_acceptance_receipt_boundary_rows")
                receipt_summary = metadata.get("adapter_acceptance_receipt_contract_summary")
                expected_receipt_summary = [
                    "proof-only",
                    "enablement-review-required",
                    "account-access-not-authorized",
                    "route-unlock-not-authorized",
                    "fresh-scope-review-required",
                ]
                if (
                    not isinstance(receipt_sha256, str)
                    or len(receipt_sha256) != 64
                    or any(char not in "0123456789abcdef" for char in receipt_sha256)
                    or metadata.get("adapter_acceptance_receipt_present") is not True
                    or metadata.get("adapter_acceptance_receipt_boundary_ready") is not True
                    or metadata.get("adapter_acceptance_receipt_boundary_row_count") != 5
                    or receipt_summary != expected_receipt_summary
                    or receipt_sha256 != handoff.get("adapter_acceptance_receipt_sha256")
                    or metadata.get("adapter_acceptance_receipt_present") != handoff.get("adapter_acceptance_receipt_present")
                    or receipt_rows != handoff.get("adapter_acceptance_receipt_boundary_rows")
                    or metadata.get("adapter_acceptance_receipt_boundary_row_count") != handoff.get("adapter_acceptance_receipt_boundary_row_count")
                    or metadata.get("adapter_acceptance_receipt_boundary_ready") != handoff.get("adapter_acceptance_receipt_boundary_ready")
                    or receipt_summary != handoff.get("adapter_acceptance_receipt_contract_summary")
                ):
                    raise SystemExit(f"Adapter acceptance receipt metadata diverged from handoff: {metadata}")
                expected_receipt_items = [
                    "adapter_acceptance_receipt",
                    "enablement_review_required",
                    "account_access_not_authorized",
                    "route_unlock_not_authorized",
                    "fresh_scope_review_required",
                ]
                if not isinstance(receipt_rows, list) or [row.get("item") for row in receipt_rows] != expected_receipt_items:
                    raise SystemExit(f"Adapter acceptance receipt boundary rows missing expected items: {metadata}")
                for row in receipt_rows:
                    if (
                        row.get("token_sha256") != receipt_sha256
                        or row.get("present") is not True
                        or row.get("authorizes_account_access")
                        or row.get("authorizes_route_unlock")
                        or row.get("authorizes_natural_language_routing")
                        or row.get("authorizes_personal_data_read")
                        or row.get("authorizes_side_effect")
                        or row.get("authorizes_approval")
                        or row.get("authorizes_model_call")
                        or row.get("authorizes_tool_execution")
                        or row.get("authorizes_external_service")
                        or row.get("authorizes_execution")
                        or row.get("authorizes_completion_claim")
                        or row.get("approval_granted")
                        or row.get("reusable_for_other_scope")
                    ):
                        raise SystemExit(f"Adapter acceptance receipt boundary row authorized too much: {metadata}")
                if (
                    metadata.get("adapter_acceptance_receipt_authorizes_account_access")
                    or metadata.get("adapter_acceptance_receipt_authorizes_route_unlock")
                    or metadata.get("adapter_acceptance_receipt_authorizes_natural_language_routing")
                    or metadata.get("adapter_acceptance_receipt_authorizes_personal_data_read")
                    or metadata.get("adapter_acceptance_receipt_authorizes_side_effect")
                    or metadata.get("adapter_acceptance_receipt_authorizes_approval")
                    or metadata.get("adapter_acceptance_receipt_authorizes_model_call")
                    or metadata.get("adapter_acceptance_receipt_authorizes_tool_execution")
                    or metadata.get("adapter_acceptance_receipt_authorizes_external_service")
                    or metadata.get("adapter_acceptance_receipt_authorizes_execution")
                    or metadata.get("adapter_acceptance_receipt_authorizes_completion_claim")
                    or metadata.get("adapter_acceptance_receipt_approval_granted")
                    or metadata.get("adapter_acceptance_receipt_reusable_for_other_scope")
                ):
                    raise SystemExit(f"Adapter acceptance receipt flat authority flags must stay false: {metadata}")
                handoff_contract = handoff.get("row_contract") or {}
                handoff_rules = handoff.get("proof_rules") or {}
                handoff_boundaries = handoff.get("boundaries") or {}
                if (
                    metadata.get("adapter_acceptance_handoff_ready") is not True
                    or handoff.get("adapter_acceptance_handoff_ready") is not True
                    or metadata.get("adapter_acceptance_handoff_ready") != handoff.get("handoff_ready")
                    or handoff.get("handoff_ready") is not True
                    or metadata.get("adapter_acceptance_ready_for_operator") != handoff.get("ready_for_operator")
                    or metadata.get("adapter_acceptance_ready_for_operator") is not True
                    or handoff.get("ready_for_operator") is not True
                    or metadata.get("adapter_acceptance_state_changed") is not False
                    or handoff.get("state_changed") is not False
                    or metadata.get("adapter_acceptance_changed") != []
                    or handoff.get("changed") != []
                    or metadata.get("adapter_acceptance_content_in_handoff") is not True
                    or handoff.get("content_in_handoff") is not True
                    or metadata.get("adapter_acceptance_authorizes_execution") is not False
                    or metadata.get("adapter_acceptance_authorizes_completion_claim") is not False
                    or metadata.get("adapter_acceptance_approval_granted") is not False
                    or handoff.get("authorizes_execution") is not False
                    or handoff.get("authorizes_completion_claim") is not False
                    or handoff.get("approval_granted") is not False
                    or metadata.get("adapter_acceptance_boundaries") != handoff_boundaries
                    or metadata.get("adapter_acceptance_next_safe_commands") != handoff.get("next_safe_commands")
                    or not metadata.get("adapter_acceptance_next_safe_commands")
                    or handoff.get("connector") != metadata.get("connector")
                    or handoff.get("requested_connector") != metadata.get("requested_connector")
                    or handoff.get("adapter_name") != metadata.get("adapter_name")
                    or handoff.get("acceptance_state") != metadata.get("acceptance_state")
                    or handoff.get("adapter_acceptance_passed") != metadata.get("adapter_acceptance_passed")
                    or handoff.get("passed_cases") != metadata.get("passed_cases")
                    or handoff.get("total_cases") != metadata.get("total_cases")
                    or handoff.get("case_results") != metadata.get("case_results")
                    or handoff_contract.get("sample_row_schema") != metadata.get("sample_row_schema")
                    or handoff_contract.get("sample_row_limit") != metadata.get("sample_row_limit")
                    or handoff_contract.get("blocked_payload_fields") != metadata.get("blocked_payload_fields")
                    or handoff_contract.get("all_case_row_contracts_present") != metadata.get("all_case_row_contracts_present")
                    or handoff.get("requires_enablement_gate") != metadata.get("requires_enablement_gate")
                    or handoff.get("requires_execution_health_report") != metadata.get("requires_execution_health_report")
                    or handoff.get("next_route_lock_requires_fresh_review") is not True
                    or handoff_rules.get("metadata_happy_path_requires_two_fake_rows") is not True
                    or handoff_rules.get("full_content_must_be_blocked") is not True
                    or handoff_rules.get("side_effect_must_be_blocked") is not True
                    or handoff_rules.get("missing_scope_must_be_blocked") is not True
                    or handoff_boundaries.get("natural_language_routing_enabled")
                    or handoff_boundaries.get("adapter_default_state") != "disabled"
                    or not handoff_boundaries.get("real_adapter_call_skipped")
                    or handoff_boundaries.get("calls_external_service")
                    or handoff_boundaries.get("reads_personal_data")
                    or handoff_boundaries.get("executes_side_effect")
                    or handoff_boundaries.get("writes_memory")
                    or handoff_boundaries.get("authorizes_account_access")
                    or handoff_boundaries.get("authorizes_route_unlock")
                    or handoff_boundaries.get("authorizes_natural_language_routing")
                    or handoff_boundaries.get("authorizes_personal_data_read")
                    or handoff_boundaries.get("authorizes_side_effect")
                    or handoff_boundaries.get("authorizes_approval")
                    or handoff_boundaries.get("authorizes_model_call")
                    or handoff_boundaries.get("authorizes_tool_execution")
                    or handoff_boundaries.get("authorizes_external_service")
                    or handoff_boundaries.get("authorizes_execution")
                    or handoff_boundaries.get("authorizes_completion_claim")
                    or handoff_boundaries.get("approval_granted")
                    or handoff_boundaries.get("reusable_for_other_scope")
                ):
                    raise SystemExit(f"Adapter acceptance handoff diverged from flat metadata or safety contract: {metadata}")
                if metadata.get("acceptance_state") != "ADAPTER_ACCEPTANCE_PASSED" or metadata.get("passed_cases") != 4:
                    raise SystemExit(f"Integration adapter acceptance should pass all disabled adapter proof cases: {metadata}")
                if metadata.get("sample_row_schema") != ["id", "timestamp", "label", "source"] or metadata.get("sample_row_limit") != 2 or metadata.get("all_case_row_contracts_present") is not True:
                    raise SystemExit(f"Integration adapter acceptance missed row schema contract metadata: {metadata}")
                if metadata.get("natural_language_routing_enabled"):
                    raise SystemExit(f"Integration adapter acceptance should keep NL routing disabled: {metadata}")
                blocked_handoff = personal_tools._adapter_acceptance_handoff_payload(
                    connector="email",
                    requested_connector="email",
                    adapter_name="email_metadata_adapter",
                    acceptance_state="ADAPTER_ACCEPTANCE_BLOCKED",
                    case_results=[
                        {
                            "name": "synthetic blocked case",
                            "passed": False,
                            "probe_state": "ADAPTER_PROBE_BLOCKED",
                            "sample_row_count": 0,
                            "sample_row_schema": ["id", "timestamp", "label", "source"],
                            "sample_row_limit": 2,
                            "sample_rows_bounded": False,
                            "blocked_payload_fields": ["body"],
                            "missing_fields": ["target/source"],
                            "full_content_blocked": True,
                            "side_effect_blocked": False,
                        }
                    ],
                    passed_cases=0,
                    sample_row_schema=["id", "timestamp", "label", "source"],
                    sample_row_limit=2,
                    blocked_payload_fields=["body"],
                    all_case_row_contracts_present=True,
                )
                blocked_boundaries = blocked_handoff.get("boundaries") or {}
                blocked_receipt_sha256 = blocked_handoff.get("adapter_acceptance_receipt_sha256")
                blocked_receipt_rows = blocked_handoff.get("adapter_acceptance_receipt_boundary_rows")
                if (
                    blocked_handoff.get("handoff_ready") is not True
                    or blocked_handoff.get("ready_for_operator") is not True
                    or blocked_handoff.get("adapter_acceptance_passed") is not False
                    or blocked_handoff.get("acceptance_state") != "ADAPTER_ACCEPTANCE_BLOCKED"
                    or not isinstance(blocked_receipt_sha256, str)
                    or len(blocked_receipt_sha256) != 64
                    or blocked_handoff.get("adapter_acceptance_receipt_present") is not True
                    or blocked_handoff.get("adapter_acceptance_receipt_boundary_ready") is not True
                    or blocked_handoff.get("adapter_acceptance_receipt_boundary_row_count") != 5
                    or not isinstance(blocked_receipt_rows, list)
                    or any(row.get("authorizes_account_access") or row.get("authorizes_route_unlock") or row.get("authorizes_execution") for row in blocked_receipt_rows)
                    or blocked_handoff.get("authorizes_execution")
                    or blocked_handoff.get("authorizes_completion_claim")
                    or blocked_handoff.get("approval_granted")
                    or blocked_boundaries.get("natural_language_routing_enabled")
                    or blocked_boundaries.get("adapter_default_state") != "disabled"
                    or not blocked_boundaries.get("real_adapter_call_skipped")
                    or blocked_boundaries.get("calls_external_service")
                    or blocked_boundaries.get("reads_personal_data")
                    or blocked_boundaries.get("executes_side_effect")
                    or blocked_boundaries.get("writes_memory")
                    or blocked_boundaries.get("authorizes_account_access")
                    or blocked_boundaries.get("authorizes_route_unlock")
                    or blocked_boundaries.get("authorizes_natural_language_routing")
                    or blocked_boundaries.get("authorizes_personal_data_read")
                    or blocked_boundaries.get("authorizes_side_effect")
                    or blocked_boundaries.get("authorizes_approval")
                    or blocked_boundaries.get("authorizes_model_call")
                    or blocked_boundaries.get("authorizes_tool_execution")
                    or blocked_boundaries.get("authorizes_external_service")
                    or blocked_boundaries.get("authorizes_execution")
                    or blocked_boundaries.get("authorizes_completion_claim")
                    or blocked_boundaries.get("approval_granted")
                    or blocked_boundaries.get("reusable_for_other_scope")
                    or not blocked_handoff.get("next_safe_commands")
                ):
                    raise SystemExit(f"Blocked adapter acceptance handoff should remain consumable and non-authorizing: {blocked_handoff}")
            elif "adapter probe" in case:
                assert_contains(
                    result.response,
                    [
                        "Jarvis personal integration adapter probe",
                        "This is read-only",
                        "disabled connector adapter boundary",
                        "Exact probe scope",
                        "Probe result",
                        "Blocked payloads",
                        "Next required commands",
                        "Hard stops",
                    ],
                    case,
                )
                if "Next proof commands" in result.response:
                    raise SystemExit(f"{case} should render adapter handoff as next required commands, not next proof commands: {result.response}")
                metadata = result.tool_results[0].metadata
                assert_read_only_metadata(metadata, "Integration adapter probe")
                handoff = metadata.get("adapter_probe_handoff")
                if not isinstance(handoff, dict):
                    raise SystemExit(f"Expected structured adapter probe handoff: {metadata}")
                handoff_scope = handoff.get("scope") or {}
                handoff_result = handoff.get("probe_result") or {}
                handoff_contract = handoff.get("row_contract") or {}
                handoff_commands = handoff.get("next_proof_commands") or {}
                handoff_risk = handoff.get("risk") or {}
                handoff_boundaries = handoff.get("boundaries") or {}
                if (
                    metadata.get("adapter_probe_handoff_ready") is not True
                    or handoff.get("adapter_probe_handoff_ready") is not True
                    or metadata.get("adapter_probe_handoff_ready") != handoff.get("handoff_ready")
                    or handoff.get("handoff_ready") is not True
                    or metadata.get("adapter_probe_ready_for_operator") != handoff.get("ready_for_operator")
                    or metadata.get("adapter_probe_ready_for_operator") is not True
                    or handoff.get("ready_for_operator") is not True
                    or metadata.get("adapter_probe_state_changed") is not False
                    or handoff.get("state_changed") is not False
                    or metadata.get("adapter_probe_changed") != []
                    or handoff.get("changed") != []
                    or metadata.get("adapter_probe_content_in_handoff") != handoff.get("content_in_handoff")
                    or metadata.get("adapter_probe_content_in_handoff") is not True
                    or handoff.get("content_in_handoff") is not True
                    or metadata.get("adapter_probe_authorizes_execution") is not False
                    or metadata.get("adapter_probe_authorizes_completion_claim") is not False
                    or metadata.get("adapter_probe_approval_granted") is not False
                    or handoff.get("authorizes_execution") is not False
                    or handoff.get("authorizes_completion_claim") is not False
                    or handoff.get("approval_granted") is not False
                    or metadata.get("adapter_probe_boundaries") != handoff_boundaries
                    or metadata.get("adapter_probe_next_safe_commands") != handoff.get("next_safe_commands")
                    or not metadata.get("adapter_probe_next_safe_commands")
                    or handoff.get("connector") != metadata.get("connector")
                    or handoff.get("requested_connector") != metadata.get("requested_connector")
                    or handoff.get("action") != metadata.get("action")
                    or handoff.get("adapter_name") != metadata.get("adapter_name")
                    or handoff.get("fixture_name") != metadata.get("fixture_name")
                    or handoff.get("adapter_state") != metadata.get("adapter_state")
                    or handoff.get("probe_state") != metadata.get("probe_state")
                    or handoff.get("adapter_probe_ready") != metadata.get("adapter_probe_ready")
                    or handoff.get("missing_fields") != metadata.get("missing_fields")
                    or handoff_scope.get("target") != metadata.get("target")
                    or handoff_scope.get("time_range") != metadata.get("time_range")
                    or handoff_scope.get("data_level") != metadata.get("data_level")
                    or handoff_scope.get("verification") != metadata.get("verification")
                    or handoff_result.get("metadata_only") != metadata.get("metadata_only")
                    or handoff_result.get("sample_rows") != metadata.get("sample_rows")
                    or handoff_result.get("sample_row_count") != metadata.get("sample_row_count")
                    or handoff_result.get("full_content_blocked") != metadata.get("full_content_blocked")
                    or handoff_result.get("side_effect_blocked") != metadata.get("side_effect_blocked")
                    or handoff_result.get("adapter_call_skipped") is not True
                    or handoff_result.get("disabled_fixture_used") != metadata.get("adapter_probe_ready")
                    or handoff_contract.get("sample_row_schema") != metadata.get("sample_row_schema")
                    or handoff_contract.get("sample_row_schema_fields") != metadata.get("sample_row_schema_fields")
                    or handoff_contract.get("sample_row_limit") != metadata.get("sample_row_limit")
                    or handoff_contract.get("sample_rows_bounded") != metadata.get("sample_rows_bounded")
                    or handoff_contract.get("blocked_payload_fields") != metadata.get("blocked_payload_fields")
                    or handoff_commands.get("adapter_manifest") != f"integration adapter manifest: {metadata.get('connector')}"
                    or handoff_commands.get("execution_health") != "execution health report"
                    or handoff_risk.get("suggested_risk") != metadata.get("suggested_risk")
                    or handoff_risk.get("approval_required") != metadata.get("approval_required")
                    or handoff_risk.get("requires_approval") != metadata.get("requires_approval")
                    or handoff_boundaries.get("natural_language_routing_enabled")
                    or handoff_boundaries.get("adapter_default_state") != "disabled"
                    or not handoff_boundaries.get("real_adapter_call_skipped")
                    or handoff_boundaries.get("calls_external_service")
                    or handoff_boundaries.get("reads_personal_data")
                    or handoff_boundaries.get("executes_side_effect")
                    or handoff_boundaries.get("writes_memory")
                    or handoff_boundaries.get("authorizes_account_access")
                    or handoff_boundaries.get("authorizes_route_unlock")
                    or handoff_boundaries.get("authorizes_natural_language_routing")
                    or handoff_boundaries.get("authorizes_personal_data_read")
                    or handoff_boundaries.get("authorizes_side_effect")
                    or handoff_boundaries.get("authorizes_approval")
                    or handoff_boundaries.get("authorizes_model_call")
                    or handoff_boundaries.get("authorizes_tool_execution")
                    or handoff_boundaries.get("authorizes_external_service")
                    or handoff_boundaries.get("authorizes_execution")
                    or handoff_boundaries.get("authorizes_completion_claim")
                    or handoff_boundaries.get("approval_granted")
                    or handoff_boundaries.get("reusable_for_other_scope")
                    or handoff.get("next_step_requires_metadata_preview") != metadata.get("adapter_probe_ready")
                    or handoff.get("next_step_requires_adapter_acceptance") != metadata.get("adapter_probe_ready")
                    or handoff.get("next_route_lock_requires_fresh_review") is not True
                ):
                    raise SystemExit(f"Adapter probe handoff diverged from flat metadata or safety contract: {metadata}")
                if metadata.get("natural_language_routing_enabled"):
                    raise SystemExit(f"Integration adapter probe should keep NL routing disabled: {metadata}")
                if metadata.get("adapter_state") != "disabled" or not str(metadata.get("adapter_name", "")).endswith("_metadata_adapter"):
                    raise SystemExit(f"Integration adapter probe missed disabled adapter metadata: {metadata}")
                if "data metadata-only" in case and "send draft reply" not in case:
                    assert_contains(result.response, ["row schema: id, timestamp, label, source", "row limit: 2", "bounded rows: yes"], case)
                    if metadata.get("probe_state") != "ADAPTER_PROBE_READY" or metadata.get("sample_row_count") != 2:
                        raise SystemExit(f"Integration adapter probe should return fake bounded rows for metadata-only scope: {metadata}")
                    if metadata.get("sample_row_schema") != ["id", "timestamp", "label", "source"] or metadata.get("sample_row_limit") != 2 or metadata.get("sample_rows_bounded") is not True:
                        raise SystemExit(f"Integration adapter probe missed bounded row schema contract: {metadata}")
                if "data full body" in case:
                    if metadata.get("probe_state") != "ADAPTER_PROBE_BLOCKED" or metadata.get("sample_row_count") != 0 or not metadata.get("full_content_blocked"):
                        raise SystemExit(f"Integration adapter probe should block full-content scope: {metadata}")
                    if metadata.get("adapter_probe_handoff_ready") is not True or metadata.get("adapter_probe_ready"):
                        raise SystemExit(f"Blocked full-content adapter probe should keep handoff ready but probe blocked: {metadata}")
                if "send draft reply" in case:
                    if metadata.get("probe_state") != "ADAPTER_PROBE_BLOCKED" or metadata.get("sample_row_count") != 0 or not metadata.get("side_effect_blocked"):
                        raise SystemExit(f"Integration adapter probe should block side-effect scope: {metadata}")
                    if metadata.get("adapter_probe_handoff_ready") is not True or metadata.get("adapter_probe_ready"):
                        raise SystemExit(f"Blocked side-effect adapter probe should keep handoff ready but probe blocked: {metadata}")
            elif "dry run" in case:
                assert_contains(
                    result.response,
                    [
                        "Jarvis personal integration dry-run contract",
                        "This is read-only",
                        "Dry-run scope",
                        "Harness receipt",
                        "Known connector boundaries",
                        "One-shot approval seed",
                        "metadata row schema: id, timestamp, label, source",
                        "metadata row limit: 2",
                        "blocked payload fields:",
                        "Hard stops",
                    ],
                    case,
                )
                metadata = result.tool_results[0].metadata
                assert_read_only_metadata(metadata, "Integration dry-run contract")
                handoff = metadata.get("dry_run_handoff")
                if not isinstance(handoff, dict):
                    raise SystemExit(f"Expected structured dry-run handoff: {metadata}")
                handoff_scope = handoff.get("scope") or {}
                handoff_receipt = handoff.get("harness_receipt") or {}
                handoff_approval = handoff.get("one_shot_approval_seed") or {}
                handoff_contract = handoff.get("metadata_row_contract") or {}
                handoff_risk = handoff.get("risk") or {}
                handoff_boundaries = handoff.get("boundaries") or {}
                if (
                    metadata.get("dry_run_handoff_ready") is not True
                    or handoff.get("dry_run_handoff_ready") is not True
                    or metadata.get("dry_run_handoff_ready") != handoff.get("handoff_ready")
                    or metadata.get("dry_run_ready_for_operator") is not True
                    or handoff.get("ready_for_operator") is not True
                    or metadata.get("dry_run_state_changed") is not False
                    or handoff.get("state_changed") is not False
                    or metadata.get("dry_run_changed") != []
                    or handoff.get("changed") != []
                    or metadata.get("dry_run_content_in_handoff") is not True
                    or handoff.get("content_in_handoff") is not True
                    or metadata.get("dry_run_authorizes_execution") is not False
                    or metadata.get("dry_run_authorizes_completion_claim") is not False
                    or metadata.get("dry_run_approval_granted") is not False
                    or handoff.get("authorizes_execution") is not False
                    or handoff.get("authorizes_completion_claim") is not False
                    or handoff.get("approval_granted") is not False
                    or metadata.get("dry_run_boundaries") != handoff_boundaries
                    or metadata.get("dry_run_next_safe_commands") != handoff.get("next_safe_commands")
                    or set((handoff.get("next_commands") or {}).values()) != set(metadata.get("dry_run_next_safe_commands") or [])
                    or handoff.get("connector") != metadata.get("connector")
                    or handoff.get("requested_connector") != metadata.get("requested_connector")
                    or handoff.get("action") != metadata.get("action")
                    or handoff.get("execution_state") != metadata.get("execution_state")
                    or handoff.get("dry_run_ready") != metadata.get("dry_run_ready")
                    or handoff.get("missing_fields") != metadata.get("missing_fields")
                    or handoff_scope.get("target") != metadata.get("target")
                    or handoff_scope.get("time_range") != metadata.get("time_range")
                    or handoff_scope.get("data_level") != metadata.get("data_level")
                    or handoff_scope.get("verification") != metadata.get("verification")
                    or handoff_receipt.get("future_tool") != f"future_{metadata.get('connector')}_connector"
                    or handoff_receipt.get("post_run_verifier") != (metadata.get("verification") or "<required before real connector use>")
                    or handoff_approval.get("approval_required_for_real_use") != metadata.get("requires_approval")
                    or handoff_approval.get("one_shot") is not True
                    or handoff_approval.get("cannot_transfer_scope") is not True
                    or handoff_contract.get("required") != metadata.get("metadata_row_contract_required")
                    or handoff_contract.get("ready") != metadata.get("metadata_row_contract_ready")
                    or handoff_contract.get("sample_row_schema") != metadata.get("sample_row_schema")
                    or handoff_contract.get("sample_row_limit") != metadata.get("sample_row_limit")
                    or handoff_contract.get("blocked_payload_fields") != metadata.get("blocked_payload_fields")
                    or handoff_risk.get("suggested_risk") != metadata.get("suggested_risk")
                    or handoff_risk.get("approval_required") != metadata.get("approval_required")
                    or handoff_risk.get("requires_approval") != metadata.get("requires_approval")
                    or handoff_boundaries.get("natural_language_routing_enabled")
                    or handoff_boundaries.get("adapter_default_state") != "disabled"
                    or handoff_boundaries.get("calls_external_service")
                    or handoff_boundaries.get("reads_personal_data")
                    or handoff_boundaries.get("executes_side_effect")
                    or handoff_boundaries.get("writes_memory")
                    or handoff_boundaries.get("authorizes_account_access")
                    or handoff_boundaries.get("authorizes_route_unlock")
                    or handoff_boundaries.get("authorizes_natural_language_routing")
                    or handoff_boundaries.get("authorizes_personal_data_read")
                    or handoff_boundaries.get("authorizes_side_effect")
                    or handoff_boundaries.get("authorizes_approval")
                    or handoff_boundaries.get("authorizes_model_call")
                    or handoff_boundaries.get("authorizes_tool_execution")
                    or handoff_boundaries.get("authorizes_external_service")
                    or handoff_boundaries.get("authorizes_execution")
                    or handoff_boundaries.get("authorizes_completion_claim")
                    or handoff_boundaries.get("approval_granted")
                    or handoff_boundaries.get("reusable_for_other_scope")
                    or handoff.get("next_step_requires_runbook") != metadata.get("dry_run_ready")
                    or handoff.get("next_step_requires_promotion_gate") != metadata.get("dry_run_ready")
                    or handoff.get("next_route_lock_requires_fresh_review") is not True
                ):
                    raise SystemExit(f"Dry-run handoff diverged from flat metadata or safety contract: {metadata}")
                if "send draft reply" in case:
                    assert_contains(result.response, ["Execution state: DRY_RUN_READY", "Suggested risk: EXTERNAL_SIDE_EFFECT", "approval required for real use: yes"], case)
                    if metadata.get("execution_state") != "DRY_RUN_READY" or not metadata.get("dry_run_ready"):
                        raise SystemExit(f"Expected ready dry-run metadata: {metadata}")
                    if metadata.get("metadata_row_contract_required") or metadata.get("metadata_row_contract_ready"):
                        raise SystemExit(f"Draft-only dry run should not require metadata row contract: {metadata}")
                if "search mailbox metadata" in case:
                    assert_contains(result.response, ["Execution state: DRY_RUN_READY", "metadata row contract: ready"], case)
                    if (
                        metadata.get("execution_state") != "DRY_RUN_READY"
                        or not metadata.get("dry_run_ready")
                        or metadata.get("metadata_row_contract_required") is not True
                        or metadata.get("metadata_row_contract_ready") is not True
                        or metadata.get("sample_row_schema") != ["id", "timestamp", "label", "source"]
                        or metadata.get("sample_row_limit") != 2
                    ):
                        raise SystemExit(f"Expected metadata row contract proof for metadata dry run: {metadata}")
                if "logged-in page" in case:
                    assert_contains(result.response, ["Execution state: HELD_FOR_SCOPE", "missing fields:", "verification target"], case)
                    if metadata.get("execution_state") != "HELD_FOR_SCOPE" or metadata.get("dry_run_ready"):
                        raise SystemExit(f"Expected held dry-run metadata: {metadata}")
            elif "runbook" in case:
                assert_contains(
                    result.response,
                    [
                        "Jarvis personal integration runbook",
                        "This is read-only",
                        "Runbook readiness",
                        "Scope lock",
                        "Preflight steps",
                        "Approval gate",
                        "Execution plan",
                        "Verification steps",
                        "Rollback and recovery",
                        "Known connector boundaries",
                        "Hard stops",
                    ],
                    case,
                )
                metadata = result.tool_results[0].metadata
                assert_read_only_metadata(metadata, "Integration runbook")
                handoff = metadata.get("runbook_handoff")
                if not isinstance(handoff, dict):
                    raise SystemExit(f"Expected structured runbook handoff: {metadata}")
                handoff_scope = handoff.get("scope_lock") or {}
                handoff_preflight = handoff.get("preflight") or {}
                handoff_approval = handoff.get("approval_gate") or {}
                handoff_execution = handoff.get("execution_plan") or {}
                handoff_verification = handoff.get("verification") or {}
                handoff_recovery = handoff.get("rollback_and_recovery") or {}
                handoff_contract = handoff.get("metadata_row_contract") or {}
                handoff_risk = handoff.get("risk") or {}
                handoff_commands = handoff.get("next_commands") or {}
                handoff_boundaries = handoff.get("boundaries") or {}
                if (
                    metadata.get("runbook_handoff_ready") is not True
                    or handoff.get("runbook_handoff_ready") is not True
                    or metadata.get("runbook_handoff_ready") != handoff.get("handoff_ready")
                    or metadata.get("runbook_ready_for_operator") is not True
                    or handoff.get("ready_for_operator") is not True
                    or metadata.get("runbook_state_changed") is not False
                    or handoff.get("state_changed") is not False
                    or metadata.get("runbook_changed") != []
                    or handoff.get("changed") != []
                    or metadata.get("runbook_content_in_handoff") is not True
                    or handoff.get("content_in_handoff") is not True
                    or metadata.get("runbook_authorizes_execution") is not False
                    or metadata.get("runbook_authorizes_completion_claim") is not False
                    or metadata.get("runbook_approval_granted") is not False
                    or handoff.get("authorizes_execution") is not False
                    or handoff.get("authorizes_completion_claim") is not False
                    or handoff.get("approval_granted") is not False
                    or metadata.get("runbook_boundaries") != handoff_boundaries
                    or metadata.get("runbook_next_safe_commands") != handoff.get("next_safe_commands")
                    or set((handoff.get("next_commands") or {}).values()) != set(metadata.get("runbook_next_safe_commands") or [])
                    or handoff.get("connector") != metadata.get("connector")
                    or handoff.get("requested_connector") != metadata.get("requested_connector")
                    or handoff.get("action") != metadata.get("action")
                    or handoff.get("readiness") != metadata.get("readiness")
                    or handoff.get("ready_for_dry_run") != (metadata.get("readiness") == "READY_FOR_DRY_RUN")
                    or handoff.get("missing_fields") != metadata.get("missing_fields")
                    or handoff_scope.get("target") != metadata.get("target")
                    or handoff_scope.get("time_range") != metadata.get("time_range")
                    or handoff_scope.get("data_level") != metadata.get("data_level")
                    or handoff_scope.get("verification") != metadata.get("verification")
                    or handoff_scope.get("rollback") != metadata.get("rollback")
                    or handoff_scope.get("verifier") != (metadata.get("verification") or "<required before real connector use>")
                    or handoff_scope.get("rollback_plan") is None
                    or handoff_preflight.get("connector_risk_declared") is not True
                    or handoff_preflight.get("exact_scope_required") is not True
                    or handoff_preflight.get("fresh_runbook_required_if_request_changes") is not True
                    or handoff_preflight.get("metadata_row_contract_required") != metadata.get("metadata_row_contract_required")
                    or handoff_preflight.get("metadata_row_contract_ready") != metadata.get("metadata_row_contract_ready")
                    or handoff_preflight.get("prefer_metadata_or_draft_only") is not True
                    or handoff_approval.get("required_for_real_execution") != metadata.get("requires_approval")
                    or handoff_approval.get("one_shot") is not True
                    or handoff_approval.get("cannot_transfer_scope") is not True
                    or handoff_execution.get("future_tool") != f"future_{metadata.get('connector')}_connector"
                    or handoff_verification.get("primary_verifier") != (metadata.get("verification") or "<required before real connector use>")
                    or handoff_verification.get("recent_tool_runs_required") is not True
                    or handoff_verification.get("runtime_trace_required") is not True
                    or handoff_verification.get("verification_receipt_required_before_completion") is not True
                    or handoff_recovery.get("rollback_plan") != handoff_scope.get("rollback_plan")
                    or handoff_recovery.get("recovery_packet_required_on_verification_failure") is not True
                    or handoff_recovery.get("corrective_side_effect_requires_fresh_approval") is not True
                    or handoff_contract.get("required") != metadata.get("metadata_row_contract_required")
                    or handoff_contract.get("ready") != metadata.get("metadata_row_contract_ready")
                    or handoff_contract.get("sample_row_schema") != metadata.get("sample_row_schema")
                    or handoff_contract.get("sample_row_limit") != metadata.get("sample_row_limit")
                    or handoff_contract.get("blocked_payload_fields") != metadata.get("blocked_payload_fields")
                    or handoff_risk.get("suggested_risk") != metadata.get("suggested_risk")
                    or handoff_risk.get("approval_required") != metadata.get("approval_required")
                    or handoff_risk.get("requires_approval") != metadata.get("requires_approval")
                    or handoff_risk.get("rollback_required") != metadata.get("rollback_required")
                    or handoff_commands.get("dry_run_contract") is None
                    or handoff_commands.get("scope_packet") != f"integration scope packet: {metadata.get('connector')} -> {metadata.get('action')}"
                    or handoff_boundaries.get("natural_language_routing_enabled")
                    or handoff_boundaries.get("adapter_default_state") != "disabled"
                    or handoff_boundaries.get("calls_external_service")
                    or handoff_boundaries.get("reads_personal_data")
                    or handoff_boundaries.get("executes_side_effect")
                    or handoff_boundaries.get("writes_memory")
                    or handoff_boundaries.get("authorizes_account_access")
                    or handoff_boundaries.get("authorizes_route_unlock")
                    or handoff_boundaries.get("authorizes_natural_language_routing")
                    or handoff_boundaries.get("authorizes_personal_data_read")
                    or handoff_boundaries.get("authorizes_side_effect")
                    or handoff_boundaries.get("authorizes_approval")
                    or handoff_boundaries.get("authorizes_model_call")
                    or handoff_boundaries.get("authorizes_tool_execution")
                    or handoff_boundaries.get("authorizes_external_service")
                    or handoff_boundaries.get("authorizes_execution")
                    or handoff_boundaries.get("authorizes_completion_claim")
                    or handoff_boundaries.get("approval_granted")
                    or handoff_boundaries.get("reusable_for_other_scope")
                    or handoff.get("next_step_requires_dry_run_contract") != (metadata.get("readiness") == "READY_FOR_DRY_RUN")
                    or handoff.get("next_step_requires_promotion_gate") != (metadata.get("readiness") == "READY_FOR_DRY_RUN")
                    or handoff.get("next_route_lock_requires_fresh_review") is not True
                ):
                    raise SystemExit(f"Runbook handoff diverged from flat metadata or safety contract: {metadata}")
                if "send draft reply" in case:
                    assert_contains(
                        result.response,
                        [
                            "Runbook readiness: READY_FOR_DRY_RUN",
                            "Suggested risk: EXTERNAL_SIDE_EFFECT",
                            "approval required before real connector use",
                            "rollback/cancel path: discard draft",
                        ],
                        case,
                    )
                    if metadata.get("readiness") != "READY_FOR_DRY_RUN" or metadata.get("missing_fields") or not metadata.get("rollback_required"):
                        raise SystemExit(f"Expected ready runbook metadata with rollback required: {metadata}")
                if "search mailbox metadata" in case:
                    assert_contains(
                        result.response,
                        [
                            "Runbook readiness: READY_FOR_DRY_RUN",
                            "Metadata row contract: ready",
                            "schema id, timestamp, label, source",
                            "row limit 2",
                            "Full private payload fields stay blocked",
                        ],
                        case,
                    )
                    if (
                        metadata.get("readiness") != "READY_FOR_DRY_RUN"
                        or metadata.get("metadata_row_contract_required") is not True
                        or metadata.get("metadata_row_contract_ready") is not True
                        or metadata.get("sample_row_schema") != ["id", "timestamp", "label", "source"]
                        or metadata.get("sample_row_limit") != 2
                    ):
                        raise SystemExit(f"Expected runbook metadata row contract proof: {metadata}")
                if "logged-in page" in case:
                    assert_contains(result.response, ["Runbook readiness: HELD_FOR_SCOPE", "missing fields:", "target/source", "verification target"], case)
                    if metadata.get("readiness") != "HELD_FOR_SCOPE" or not metadata.get("missing_fields"):
                        raise SystemExit(f"Expected held runbook metadata: {metadata}")
            elif "promotion gate" in case:
                assert_contains(
                    result.response,
                    [
                        "Jarvis personal integration promotion gate",
                        "This is read-only",
                        "Promotion verdict",
                        "Evidence checked",
                        "Required implementation invariants",
                        "Smoke tests required",
                        "Allowed next build move",
                        "Known connector boundaries",
                        "Hard stops",
                    ],
                    case,
                )
                metadata = result.tool_results[0].metadata
                assert_read_only_metadata(metadata, "Integration promotion gate")
                handoff = metadata.get("promotion_handoff")
                if not isinstance(handoff, dict):
                    raise SystemExit(f"Expected structured promotion handoff: {metadata}")
                handoff_evidence = handoff.get("evidence") or {}
                handoff_contract = handoff.get("metadata_row_contract") or {}
                handoff_risk = handoff.get("risk") or {}
                handoff_invariants = handoff.get("implementation_invariants") or {}
                handoff_smoke = handoff.get("smoke_tests_required") or {}
                handoff_commands = handoff.get("next_commands") or {}
                handoff_boundaries = handoff.get("boundaries") or {}
                if (
                    metadata.get("promotion_handoff_ready") is not True
                    or handoff.get("promotion_handoff_ready") is not True
                    or metadata.get("promotion_handoff_ready") != handoff.get("handoff_ready")
                    or metadata.get("promotion_ready_for_operator") is not True
                    or handoff.get("ready_for_operator") is not True
                    or metadata.get("promotion_state_changed") is not False
                    or handoff.get("state_changed") is not False
                    or metadata.get("promotion_changed") != []
                    or handoff.get("changed") != []
                    or metadata.get("promotion_content_in_handoff") is not True
                    or handoff.get("content_in_handoff") is not True
                    or metadata.get("promotion_authorizes_execution") is not False
                    or metadata.get("promotion_authorizes_completion_claim") is not False
                    or metadata.get("promotion_approval_granted") is not False
                    or handoff.get("authorizes_execution") is not False
                    or handoff.get("authorizes_completion_claim") is not False
                    or handoff.get("approval_granted") is not False
                    or metadata.get("promotion_boundaries") != handoff_boundaries
                    or metadata.get("promotion_next_safe_commands") != handoff.get("next_safe_commands")
                    or set((handoff.get("next_commands") or {}).values()) != set(metadata.get("promotion_next_safe_commands") or [])
                    or handoff.get("connector") != metadata.get("connector")
                    or handoff.get("requested_connector") != metadata.get("requested_connector")
                    or handoff.get("action") != metadata.get("action")
                    or handoff.get("promotion_verdict") != metadata.get("promotion_verdict")
                    or handoff.get("implementation_allowed") != metadata.get("implementation_allowed")
                    or handoff.get("missing_fields") != metadata.get("missing_fields")
                    or handoff_evidence.get("target") != metadata.get("target")
                    or handoff_evidence.get("time_range") != metadata.get("time_range")
                    or handoff_evidence.get("data_level") != metadata.get("data_level")
                    or handoff_evidence.get("verification") != metadata.get("verification")
                    or handoff_evidence.get("rollback") != metadata.get("rollback")
                    or handoff_evidence.get("tests") != metadata.get("tests")
                    or handoff_evidence.get("audit") != metadata.get("audit")
                    or handoff_contract.get("required") != metadata.get("metadata_row_contract_required")
                    or handoff_contract.get("ready") != metadata.get("metadata_row_contract_ready")
                    or handoff_contract.get("sample_row_schema") != metadata.get("sample_row_schema")
                    or handoff_contract.get("sample_row_limit") != metadata.get("sample_row_limit")
                    or handoff_contract.get("blocked_payload_fields") != metadata.get("blocked_payload_fields")
                    or handoff_risk.get("suggested_risk") != metadata.get("suggested_risk")
                    or handoff_risk.get("approval_required") != metadata.get("approval_required")
                    or handoff_risk.get("requires_approval") != metadata.get("requires_approval")
                    or handoff_risk.get("rollback_required") != metadata.get("rollback_required")
                    or handoff_invariants.get("narrow_tool_registry_risk_required") is not True
                    or handoff_invariants.get("natural_language_routing_enabled")
                    or handoff_invariants.get("bounded_audit_summary_only") is not True
                    or handoff_invariants.get("metadata_rows_bounded_until_review") is not True
                    or handoff_invariants.get("personal_data_and_side_effects_need_approval_chain") is not True
                    or handoff_smoke.get("happy_path") != metadata.get("tests")
                    or handoff_smoke.get("blocked_scope_path") is not True
                    or handoff_smoke.get("approval_path_for_private_or_side_effect") != metadata.get("requires_approval")
                    or handoff_smoke.get("audit_path") is not True
                    or handoff_commands.get("runbook") is None
                    or handoff_commands.get("implementation_spec") is None
                    or handoff_commands.get("preflight_contract") is None
                    or handoff_boundaries.get("natural_language_routing_enabled")
                    or handoff_boundaries.get("adapter_default_state") != "disabled"
                    or handoff_boundaries.get("calls_external_service")
                    or handoff_boundaries.get("reads_personal_data")
                    or handoff_boundaries.get("executes_side_effect")
                    or handoff_boundaries.get("writes_memory")
                    or handoff_boundaries.get("authorizes_account_access")
                    or handoff_boundaries.get("authorizes_route_unlock")
                    or handoff_boundaries.get("authorizes_natural_language_routing")
                    or handoff_boundaries.get("authorizes_personal_data_read")
                    or handoff_boundaries.get("authorizes_side_effect")
                    or handoff_boundaries.get("authorizes_approval")
                    or handoff_boundaries.get("authorizes_model_call")
                    or handoff_boundaries.get("authorizes_tool_execution")
                    or handoff_boundaries.get("authorizes_external_service")
                    or handoff_boundaries.get("authorizes_execution")
                    or handoff_boundaries.get("authorizes_completion_claim")
                    or handoff_boundaries.get("approval_granted")
                    or handoff_boundaries.get("reusable_for_other_scope")
                    or handoff.get("next_step_requires_implementation_spec") != metadata.get("implementation_allowed")
                    or handoff.get("next_step_requires_preflight_contract") != metadata.get("implementation_allowed")
                    or handoff.get("next_route_lock_requires_fresh_review") is not True
                ):
                    raise SystemExit(f"Promotion handoff diverged from flat metadata or safety contract: {metadata}")
                if "send draft reply" in case:
                    assert_contains(
                        result.response,
                        [
                            "Promotion verdict: READY_FOR_APPROVAL_GATED_IMPLEMENTATION_SPEC",
                            "Suggested risk: EXTERNAL_SIDE_EFFECT",
                            "Approval required for real use: yes",
                            "smoke test plan: blocked send smoke",
                            "audit trail plan: tool run receipt",
                            "missing blockers: none",
                        ],
                        case,
                    )
                    if (
                        metadata.get("promotion_verdict") != "READY_FOR_APPROVAL_GATED_IMPLEMENTATION_SPEC"
                        or metadata.get("missing_fields")
                        or not metadata.get("implementation_allowed")
                        or not metadata.get("rollback_required")
                    ):
                        raise SystemExit(f"Expected approval-gated promotion metadata: {metadata}")
                if "search mailbox metadata" in case:
                    assert_contains(
                        result.response,
                        [
                            "Promotion verdict: READY_FOR_APPROVAL_GATED_IMPLEMENTATION_SPEC",
                            "metadata row contract: ready",
                            "metadata row schema: id, timestamp, label, source",
                            "metadata row limit: 2",
                            "Metadata-only adapters must keep rows limited to id, timestamp, label, source",
                        ],
                        case,
                    )
                    if (
                        metadata.get("promotion_verdict") != "READY_FOR_APPROVAL_GATED_IMPLEMENTATION_SPEC"
                        or metadata.get("metadata_row_contract_required") is not True
                        or metadata.get("metadata_row_contract_ready") is not True
                        or metadata.get("sample_row_schema") != ["id", "timestamp", "label", "source"]
                        or metadata.get("sample_row_limit") != 2
                    ):
                        raise SystemExit(f"Expected promotion metadata row contract proof: {metadata}")
                if "logged-in page" in case:
                    assert_contains(result.response, ["Promotion verdict: PROMOTION_BLOCKED", "missing blockers:", "target/source", "smoke test plan", "audit trail plan"], case)
                    if metadata.get("promotion_verdict") != "PROMOTION_BLOCKED" or not metadata.get("missing_fields") or metadata.get("implementation_allowed"):
                        raise SystemExit(f"Expected blocked promotion metadata: {metadata}")
            elif "implementation spec" in case:
                assert_contains(
                    result.response,
                    [
                        "Jarvis personal integration implementation spec",
                        "This is read-only",
                        "Implementation contract",
                        "Required arguments",
                        "Execution gates",
                        "Smoke tests required",
                        "Audit contract",
                        "Known connector boundaries",
                        "Missing blockers",
                        "Hard stops",
                    ],
                    case,
                )
                metadata = result.tool_results[0].metadata
                assert_read_only_metadata(metadata, "Integration implementation spec")
                handoff = metadata.get("implementation_spec_handoff")
                if not isinstance(handoff, dict):
                    raise SystemExit(f"Expected structured implementation spec handoff: {metadata}")
                handoff_contract = handoff.get("implementation_contract") or {}
                handoff_scope = handoff.get("scope") or {}
                handoff_proof = handoff.get("proof") or {}
                handoff_metadata_contract = handoff.get("metadata_row_contract") or {}
                handoff_risk = handoff.get("risk") or {}
                handoff_audit = handoff.get("audit_contract") or {}
                handoff_commands = handoff.get("next_commands") or {}
                handoff_boundaries = handoff.get("boundaries") or {}
                if (
                    metadata.get("implementation_spec_handoff_ready") is not True
                    or handoff.get("implementation_spec_handoff_ready") is not True
                    or metadata.get("implementation_spec_handoff_ready") != handoff.get("handoff_ready")
                    or metadata.get("implementation_spec_ready_for_operator") is not True
                    or handoff.get("ready_for_operator") is not True
                    or metadata.get("implementation_spec_state_changed") is not False
                    or handoff.get("state_changed") is not False
                    or metadata.get("implementation_spec_changed") != []
                    or handoff.get("changed") != []
                    or metadata.get("implementation_spec_content_in_handoff") is not True
                    or handoff.get("content_in_handoff") is not True
                    or metadata.get("implementation_spec_authorizes_execution") is not False
                    or metadata.get("implementation_spec_authorizes_completion_claim") is not False
                    or metadata.get("implementation_spec_approval_granted") is not False
                    or handoff.get("authorizes_execution") is not False
                    or handoff.get("authorizes_completion_claim") is not False
                    or handoff.get("approval_granted") is not False
                    or metadata.get("implementation_spec_boundaries") != handoff_boundaries
                    or metadata.get("implementation_spec_next_safe_commands") != handoff.get("next_safe_commands")
                    or set((handoff.get("next_commands") or {}).values()) != set(metadata.get("implementation_spec_next_safe_commands") or [])
                    or handoff.get("connector") != metadata.get("connector")
                    or handoff.get("requested_connector") != metadata.get("requested_connector")
                    or handoff.get("action") != metadata.get("action")
                    or handoff.get("tool_name") != metadata.get("tool_name")
                    or handoff.get("spec_state") != metadata.get("spec_state")
                    or handoff.get("spec_ready_for_review") != (metadata.get("spec_state") == "SPEC_READY_FOR_REVIEW")
                    or handoff.get("implementation_spec_version") != metadata.get("implementation_spec_version")
                    or handoff.get("missing_fields") != metadata.get("missing_fields")
                    or handoff_contract.get("tool_name") != metadata.get("tool_name")
                    or handoff_contract.get("toolset") != "personal"
                    or handoff_contract.get("planner_route") != metadata.get("planner_route")
                    or handoff_contract.get("optional_status_api") != metadata.get("endpoint")
                    or handoff_contract.get("natural_language_routing_enabled")
                    or handoff_scope.get("target") != metadata.get("target")
                    or handoff_scope.get("time_range") != metadata.get("time_range")
                    or handoff_scope.get("data_level") != metadata.get("data_level")
                    or handoff_scope.get("verification") != metadata.get("verification")
                    or handoff_scope.get("rollback") != metadata.get("rollback")
                    or handoff_proof.get("tests") != metadata.get("tests")
                    or handoff_proof.get("audit") != metadata.get("audit")
                    or handoff_metadata_contract.get("required") != metadata.get("metadata_row_contract_required")
                    or handoff_metadata_contract.get("ready") != metadata.get("metadata_row_contract_ready")
                    or handoff_metadata_contract.get("sample_row_schema") != metadata.get("sample_row_schema")
                    or handoff_metadata_contract.get("sample_row_limit") != metadata.get("sample_row_limit")
                    or handoff_metadata_contract.get("blocked_payload_fields") != metadata.get("blocked_payload_fields")
                    or handoff_risk.get("suggested_risk") != metadata.get("suggested_risk")
                    or handoff_risk.get("approval_required") != metadata.get("approval_required")
                    or handoff_risk.get("requires_approval") != metadata.get("requires_approval")
                    or handoff_risk.get("rollback_required") != metadata.get("rollback_required")
                    or handoff_audit.get("implementation_spec_version") != metadata.get("implementation_spec_version")
                    or handoff_audit.get("metadata_row_schema") != metadata.get("sample_row_schema")
                    or handoff_audit.get("metadata_row_limit") != metadata.get("sample_row_limit")
                    or handoff_audit.get("blocked_payload_fields") != metadata.get("blocked_payload_fields")
                    or handoff_commands.get("promotion_gate") is None
                    or handoff_commands.get("preflight_contract") is None
                    or handoff_commands.get("enablement_gate") is None
                    or handoff_boundaries.get("natural_language_routing_enabled")
                    or handoff_boundaries.get("adapter_default_state") != "disabled"
                    or handoff_boundaries.get("calls_external_service")
                    or handoff_boundaries.get("reads_personal_data")
                    or handoff_boundaries.get("executes_side_effect")
                    or handoff_boundaries.get("writes_memory")
                    or handoff_boundaries.get("authorizes_account_access")
                    or handoff_boundaries.get("authorizes_route_unlock")
                    or handoff_boundaries.get("authorizes_natural_language_routing")
                    or handoff_boundaries.get("authorizes_personal_data_read")
                    or handoff_boundaries.get("authorizes_side_effect")
                    or handoff_boundaries.get("authorizes_approval")
                    or handoff_boundaries.get("authorizes_model_call")
                    or handoff_boundaries.get("authorizes_tool_execution")
                    or handoff_boundaries.get("authorizes_external_service")
                    or handoff_boundaries.get("authorizes_execution")
                    or handoff_boundaries.get("authorizes_completion_claim")
                    or handoff_boundaries.get("approval_granted")
                    or handoff_boundaries.get("reusable_for_other_scope")
                    or not handoff.get("next_step_requires_preflight_contract")
                    or not handoff.get("next_step_requires_enablement_gate")
                    or not handoff.get("next_route_lock_requires_fresh_review")
                ):
                    raise SystemExit(f"Expected implementation spec handoff to mirror safe metadata contract: {metadata}")
                if "send draft reply" in case:
                    assert_contains(
                        result.response,
                        [
                            "Spec state: SPEC_READY_FOR_REVIEW",
                            "Suggested risk: EXTERNAL_SIDE_EFFECT",
                            "Approval gate: required before any real connector call",
                            "tool name:",
                            "planner route:",
                            "happy path: blocked send smoke",
                            "audit metadata must include implementation_spec_version=1",
                            "Missing blockers:",
                            "- none",
                        ],
                        case,
                    )
                    if (
                        metadata.get("spec_state") != "SPEC_READY_FOR_REVIEW"
                        or metadata.get("missing_fields")
                        or metadata.get("suggested_risk") != "EXTERNAL_SIDE_EFFECT"
                        or metadata.get("implementation_spec_version") != 1
                        or metadata.get("natural_language_routing_enabled")
                        or not metadata.get("rollback_required")
                    ):
                        raise SystemExit(f"Expected ready implementation spec metadata: {metadata}")
                if "search mailbox metadata" in case:
                    assert_contains(
                        result.response,
                        [
                            "Spec state: SPEC_READY_FOR_REVIEW",
                            "metadata row contract: ready",
                            "metadata row schema: id, timestamp, label, source",
                            "metadata row limit: 2",
                            "metadata-only audit rows must use only id, timestamp, label, source",
                        ],
                        case,
                    )
                    if (
                        metadata.get("spec_state") != "SPEC_READY_FOR_REVIEW"
                        or metadata.get("metadata_row_contract_required") is not True
                        or metadata.get("metadata_row_contract_ready") is not True
                        or metadata.get("sample_row_schema") != ["id", "timestamp", "label", "source"]
                        or metadata.get("sample_row_limit") != 2
                    ):
                        raise SystemExit(f"Expected implementation spec metadata row contract proof: {metadata}")
                if "logged-in page" in case:
                    assert_contains(result.response, ["Spec state: SPEC_BLOCKED", "Missing blockers:", "target/source", "smoke test plan", "audit trail plan"], case)
                    if metadata.get("spec_state") != "SPEC_BLOCKED" or not metadata.get("missing_fields") or metadata.get("natural_language_routing_enabled"):
                        raise SystemExit(f"Expected blocked implementation spec metadata: {metadata}")
            elif "preflight contract" in case:
                assert_contains(
                    result.response,
                    [
                        "Jarvis personal integration preflight contract",
                        "This is read-only",
                        "Preflight state",
                        "Exact future tool contract",
                        "Enablement gates",
                        "Required proof before enabling",
                        "Approval packet seed",
                        "Known connector boundaries",
                        "Stop conditions",
                    ],
                    case,
                )
                metadata = result.tool_results[0].metadata
                assert_read_only_metadata(metadata, "Integration preflight contract")
                handoff = metadata.get("preflight_handoff")
                if not isinstance(handoff, dict):
                    raise SystemExit(f"Expected structured preflight handoff: {metadata}")
                handoff_scope = handoff.get("scope") or {}
                handoff_proof = handoff.get("proof") or {}
                handoff_approval = handoff.get("approval_packet") or {}
                handoff_contract = handoff.get("metadata_row_contract") or {}
                handoff_risk = handoff.get("risk") or {}
                handoff_gates = handoff.get("enablement_gates") or {}
                handoff_required_proof = handoff.get("required_proof_before_enabling") or {}
                handoff_commands = handoff.get("next_commands") or {}
                handoff_boundaries = handoff.get("boundaries") or {}
                if (
                    metadata.get("preflight_handoff_ready") is not True
                    or handoff.get("preflight_handoff_ready") is not True
                    or metadata.get("preflight_handoff_ready") != handoff.get("handoff_ready")
                    or metadata.get("preflight_ready_for_operator") is not True
                    or handoff.get("ready_for_operator") is not True
                    or metadata.get("preflight_state_changed") is not False
                    or handoff.get("state_changed") is not False
                    or metadata.get("preflight_changed") != []
                    or handoff.get("changed") != []
                    or metadata.get("preflight_content_in_handoff") is not True
                    or handoff.get("content_in_handoff") is not True
                    or metadata.get("preflight_authorizes_execution") is not False
                    or metadata.get("preflight_authorizes_completion_claim") is not False
                    or metadata.get("preflight_approval_granted") is not False
                    or handoff.get("authorizes_execution") is not False
                    or handoff.get("authorizes_completion_claim") is not False
                    or handoff.get("approval_granted") is not False
                    or metadata.get("preflight_boundaries") != handoff_boundaries
                    or metadata.get("preflight_next_safe_commands") != handoff.get("next_safe_commands")
                    or set((handoff.get("next_commands") or {}).values()) != set(metadata.get("preflight_next_safe_commands") or [])
                    or handoff.get("connector") != metadata.get("connector")
                    or handoff.get("requested_connector") != metadata.get("requested_connector")
                    or handoff.get("action") != metadata.get("action")
                    or handoff.get("tool_name") != metadata.get("tool_name")
                    or handoff.get("preflight_state") != metadata.get("preflight_state")
                    or handoff.get("preflight_ready_for_review") != (metadata.get("preflight_state") == "PREFLIGHT_READY_FOR_REVIEW")
                    or handoff.get("missing_fields") != metadata.get("missing_fields")
                    or handoff.get("exact_args") != metadata.get("exact_args")
                    or handoff_scope.get("target") != metadata.get("target")
                    or handoff_scope.get("time_range") != metadata.get("time_range")
                    or handoff_scope.get("data_level") != metadata.get("data_level")
                    or handoff_scope.get("verification") != metadata.get("verification")
                    or handoff_scope.get("rollback") != metadata.get("rollback")
                    or handoff_proof.get("tests") != metadata.get("tests")
                    or handoff_proof.get("audit") != metadata.get("audit")
                    or handoff_approval.get("one_shot") is not True
                    or handoff_approval.get("cannot_transfer_scope") is not True
                    or handoff_approval.get("required_for_real_use") != metadata.get("requires_approval")
                    or handoff_contract.get("required") != metadata.get("metadata_row_contract_required")
                    or handoff_contract.get("ready") != metadata.get("metadata_row_contract_ready")
                    or handoff_contract.get("sample_row_schema") != metadata.get("sample_row_schema")
                    or handoff_contract.get("sample_row_limit") != metadata.get("sample_row_limit")
                    or handoff_contract.get("blocked_payload_fields") != metadata.get("blocked_payload_fields")
                    or handoff_risk.get("suggested_risk") != metadata.get("suggested_risk")
                    or handoff_risk.get("approval_required") != metadata.get("approval_required")
                    or handoff_risk.get("requires_approval") != metadata.get("requires_approval")
                    or handoff_risk.get("rollback_required") != metadata.get("rollback_required")
                    or handoff_gates.get("count") != metadata.get("enablement_gates")
                    or handoff_gates.get("implementation_spec_exists") is not True
                    or handoff_gates.get("focused_smoke_required") is not True
                    or handoff_gates.get("blocked_action_smoke_required") is not True
                    or handoff_gates.get("approval_readiness_required_for_private_or_side_effect") != metadata.get("requires_approval")
                    or handoff_gates.get("audit_row_required") is not True
                    or handoff_gates.get("verification_receipt_required") is not True
                    or handoff_required_proof.get("smoke_tests") != metadata.get("tests")
                    or handoff_required_proof.get("audit_plan") != metadata.get("audit")
                    or handoff_required_proof.get("metadata_row_contract_required") != metadata.get("metadata_row_contract_required")
                    or handoff_required_proof.get("compile_command") != "python3 -m compileall -q jarvis_v2"
                    or handoff_required_proof.get("focused_personal_smoke_command") != "python3 -m jarvis_v2.scripts.smoke_test_personal"
                    or handoff_commands.get("implementation_spec") is None
                    or handoff_commands.get("enablement_gate") is None
                    or handoff_commands.get("rehearsal_receipt") is None
                    or handoff_boundaries.get("natural_language_routing_enabled")
                    or handoff_boundaries.get("adapter_default_state") != "disabled"
                    or handoff_boundaries.get("calls_external_service")
                    or handoff_boundaries.get("reads_personal_data")
                    or handoff_boundaries.get("executes_side_effect")
                    or handoff_boundaries.get("writes_memory")
                    or handoff_boundaries.get("authorizes_account_access")
                    or handoff_boundaries.get("authorizes_route_unlock")
                    or handoff_boundaries.get("authorizes_natural_language_routing")
                    or handoff_boundaries.get("authorizes_personal_data_read")
                    or handoff_boundaries.get("authorizes_side_effect")
                    or handoff_boundaries.get("authorizes_approval")
                    or handoff_boundaries.get("authorizes_model_call")
                    or handoff_boundaries.get("authorizes_tool_execution")
                    or handoff_boundaries.get("authorizes_external_service")
                    or handoff_boundaries.get("authorizes_execution")
                    or handoff_boundaries.get("authorizes_completion_claim")
                    or handoff_boundaries.get("approval_granted")
                    or handoff_boundaries.get("reusable_for_other_scope")
                    or handoff.get("next_step_requires_enablement_gate") is not True
                    or handoff.get("next_step_requires_rehearsal_receipt") is not True
                    or handoff.get("next_step_requires_proof_bundle") is not True
                    or handoff.get("next_route_lock_requires_fresh_review") is not True
                ):
                    raise SystemExit(f"Preflight handoff diverged from flat metadata or safety contract: {metadata}")
                if "send draft reply" in case:
                    assert_contains(
                        result.response,
                        [
                            "Preflight state: PREFLIGHT_READY_FOR_REVIEW",
                            "Suggested risk: EXTERNAL_SIDE_EFFECT",
                            "Approval required for real use: yes",
                            "tool name:",
                            "missing blockers: none",
                            "metadata row contract: not required for this preflight",
                            "gate 4: personal-data or side-effect route creates approval readiness, a one-shot approval packet, and approval chain proof",
                            "focused personal smoke",
                        ],
                        case,
                    )
                    if (
                        metadata.get("preflight_state") != "PREFLIGHT_READY_FOR_REVIEW"
                        or metadata.get("missing_fields")
                        or metadata.get("suggested_risk") != "EXTERNAL_SIDE_EFFECT"
                        or metadata.get("natural_language_routing_enabled")
                        or not metadata.get("rollback_required")
                        or metadata.get("enablement_gates") != 6
                        or metadata.get("metadata_row_contract_required")
                        or metadata.get("metadata_row_contract_ready")
                    ):
                        raise SystemExit(f"Expected ready preflight metadata: {metadata}")
                    exact_args = metadata.get("exact_args") or {}
                    if exact_args.get("target") != "thread 123" or exact_args.get("verification") != "confirm not sent":
                        raise SystemExit(f"Expected exact preflight args in metadata: {metadata}")
                if "search mailbox metadata" in case:
                    assert_contains(
                        result.response,
                        [
                            "Preflight state: PREFLIGHT_READY_FOR_REVIEW",
                            "metadata row contract: ready",
                            "metadata row schema: id, timestamp, label, source",
                            "metadata row limit: 2",
                            "metadata row contract for metadata-only adapters",
                        ],
                        case,
                    )
                    if (
                        metadata.get("preflight_state") != "PREFLIGHT_READY_FOR_REVIEW"
                        or metadata.get("missing_fields")
                        or metadata.get("metadata_row_contract_required") is not True
                        or metadata.get("metadata_row_contract_ready") is not True
                        or metadata.get("sample_row_schema") != ["id", "timestamp", "label", "source"]
                        or metadata.get("sample_row_limit") != 2
                        or metadata.get("natural_language_routing_enabled")
                        or metadata.get("enablement_gates") != 6
                    ):
                        raise SystemExit(f"Expected metadata preflight row-contract metadata: {metadata}")
                if "logged-in page" in case:
                    assert_contains(result.response, ["Preflight state: PREFLIGHT_BLOCKED", "missing blockers:", "target/source", "smoke test plan", "audit trail plan"], case)
                    if metadata.get("preflight_state") != "PREFLIGHT_BLOCKED" or not metadata.get("missing_fields") or metadata.get("natural_language_routing_enabled"):
                        raise SystemExit(f"Expected blocked preflight metadata: {metadata}")
            elif "enablement gate" in case:
                assert_contains(
                    result.response,
                    [
                        "Jarvis personal integration enablement gate",
                        "This is read-only",
                        "Enablement verdict",
                        "Evidence checked",
                        "Enablement decision",
                        "Required proof before enabling",
                        "Acceptance gate",
                        "Planner routing state",
                        "Known connector boundaries",
                        "Stop conditions",
                    ],
                    case,
                )
                metadata = result.tool_results[0].metadata
                assert_read_only_metadata(metadata, "Integration enablement gate")
                handoff = metadata.get("enablement_handoff")
                if not isinstance(handoff, dict):
                    raise SystemExit(f"Expected structured enablement handoff: {metadata}")
                handoff_scope = handoff.get("scope") or {}
                handoff_proof = handoff.get("proof") or {}
                handoff_acceptance = handoff.get("adapter_acceptance") or {}
                handoff_contract = handoff.get("metadata_row_contract") or {}
                handoff_risk = handoff.get("risk") or {}
                handoff_commands = handoff.get("next_commands") or {}
                handoff_boundaries = handoff.get("boundaries") or {}
                if (
                    metadata.get("enablement_handoff_ready") is not True
                    or handoff.get("enablement_handoff_ready") is not True
                    or metadata.get("enablement_handoff_ready") != handoff.get("handoff_ready")
                    or metadata.get("enablement_ready_for_operator") is not True
                    or handoff.get("ready_for_operator") is not True
                    or metadata.get("enablement_state_changed") is not False
                    or handoff.get("state_changed") is not False
                    or metadata.get("enablement_changed") != []
                    or handoff.get("changed") != []
                    or metadata.get("enablement_content_in_handoff") is not True
                    or handoff.get("content_in_handoff") is not True
                    or metadata.get("enablement_authorizes_execution") is not False
                    or metadata.get("enablement_authorizes_completion_claim") is not False
                    or metadata.get("enablement_approval_granted") is not False
                    or handoff.get("authorizes_execution") is not False
                    or handoff.get("authorizes_completion_claim") is not False
                    or handoff.get("approval_granted") is not False
                    or metadata.get("enablement_boundaries") != handoff_boundaries
                    or metadata.get("enablement_next_safe_commands") != handoff.get("next_safe_commands")
                    or set((handoff.get("next_commands") or {}).values()) != set(metadata.get("enablement_next_safe_commands") or [])
                    or handoff.get("connector") != metadata.get("connector")
                    or handoff.get("requested_connector") != metadata.get("requested_connector")
                    or handoff.get("action") != metadata.get("action")
                    or handoff.get("tool_name") != metadata.get("tool_name")
                    or handoff.get("enablement_verdict") != metadata.get("enablement_verdict")
                    or handoff.get("enablement_ready_for_review") != metadata.get("implementation_review_allowed")
                    or handoff.get("implementation_review_allowed") != metadata.get("implementation_review_allowed")
                    or handoff.get("missing_fields") != metadata.get("missing_fields")
                    or handoff_scope.get("target") != metadata.get("target")
                    or handoff_scope.get("time_range") != metadata.get("time_range")
                    or handoff_scope.get("data_level") != metadata.get("data_level")
                    or handoff_scope.get("verification") != metadata.get("verification")
                    or handoff_scope.get("rollback") != metadata.get("rollback")
                    or handoff_proof.get("tests") != metadata.get("tests")
                    or handoff_proof.get("audit") != metadata.get("audit")
                    or handoff_proof.get("acceptance") != metadata.get("acceptance")
                    or handoff_acceptance.get("required") != metadata.get("adapter_acceptance_required")
                    or handoff_acceptance.get("state") != metadata.get("adapter_acceptance_state")
                    or handoff_acceptance.get("passed") != metadata.get("adapter_acceptance_passed")
                    or handoff_acceptance.get("passed_case_count") != metadata.get("adapter_acceptance_case_count")
                    or handoff_contract.get("required") != metadata.get("metadata_row_contract_required")
                    or handoff_contract.get("ready") != metadata.get("metadata_row_contract_ready")
                    or handoff_contract.get("sample_row_schema") != metadata.get("sample_row_schema")
                    or handoff_contract.get("sample_row_limit") != metadata.get("sample_row_limit")
                    or handoff_contract.get("blocked_payload_fields") != metadata.get("blocked_payload_fields")
                    or handoff_risk.get("suggested_risk") != metadata.get("suggested_risk")
                    or handoff_risk.get("approval_required") != metadata.get("approval_required")
                    or handoff_risk.get("requires_approval") != metadata.get("requires_approval")
                    or handoff_risk.get("rollback_required") != metadata.get("rollback_required")
                    or handoff.get("enablement_gates") != metadata.get("enablement_gates")
                    or handoff_commands.get("preflight_contract") is None
                    or handoff_commands.get("rehearsal_receipt") is None
                    or handoff_commands.get("proof_bundle") is None
                    or handoff.get("next_step_requires_rehearsal_receipt") is not True
                    or handoff.get("next_step_requires_proof_bundle") is not True
                    or handoff.get("next_route_lock_requires_fresh_review") is not True
                    or handoff_boundaries.get("natural_language_routing_enabled")
                    or handoff_boundaries.get("adapter_default_state") != "disabled"
                    or handoff_boundaries.get("calls_external_service")
                    or handoff_boundaries.get("reads_personal_data")
                    or handoff_boundaries.get("executes_side_effect")
                    or handoff_boundaries.get("writes_memory")
                    or handoff_boundaries.get("authorizes_account_access")
                    or handoff_boundaries.get("authorizes_route_unlock")
                    or handoff_boundaries.get("authorizes_natural_language_routing")
                    or handoff_boundaries.get("authorizes_personal_data_read")
                    or handoff_boundaries.get("authorizes_side_effect")
                    or handoff_boundaries.get("authorizes_approval")
                    or handoff_boundaries.get("authorizes_model_call")
                    or handoff_boundaries.get("authorizes_tool_execution")
                    or handoff_boundaries.get("authorizes_external_service")
                    or handoff_boundaries.get("authorizes_execution")
                    or handoff_boundaries.get("authorizes_completion_claim")
                    or handoff_boundaries.get("approval_granted")
                    or handoff_boundaries.get("reusable_for_other_scope")
                ):
                    raise SystemExit(f"Enablement handoff diverged from flat metadata or safety contract: {metadata}")
                if metadata.get("adapter_acceptance_required"):
                    nested_acceptance_handoff = handoff_acceptance.get("handoff")
                    if not isinstance(nested_acceptance_handoff, dict) or nested_acceptance_handoff.get("acceptance_state") != metadata.get("adapter_acceptance_state"):
                        raise SystemExit(f"Enablement handoff missed nested adapter acceptance proof: {metadata}")
                elif handoff_acceptance.get("handoff") is not None:
                    raise SystemExit(f"Enablement handoff should not carry adapter acceptance proof when not required: {metadata}")
                if "send draft reply" in case:
                    assert_contains(
                        result.response,
                        [
                            "Enablement verdict: READY_FOR_APPROVAL_GATED_ENABLEMENT_REVIEW",
                            "Suggested risk: EXTERNAL_SIDE_EFFECT",
                            "Approval required for real use: required before any real personal-data read or side effect",
                            "smoke tests: blocked send smoke",
                            "audit trail: tool run receipt",
                            "acceptance gate: gate passed with receipts",
                            "metadata row contract: ready",
                            "missing blockers: none",
                            "natural-language auto-routing: disabled",
                        ],
                        case,
                    )
                    if (
                        metadata.get("enablement_verdict") != "READY_FOR_APPROVAL_GATED_ENABLEMENT_REVIEW"
                        or metadata.get("missing_fields")
                        or metadata.get("suggested_risk") != "EXTERNAL_SIDE_EFFECT"
                        or metadata.get("natural_language_routing_enabled")
                        or metadata.get("enablement_gates") != 7
                        or not metadata.get("rollback_required")
                        or not metadata.get("implementation_review_allowed")
                        or metadata.get("metadata_row_contract_required")
                        or metadata.get("metadata_row_contract_ready") is not True
                    ):
                        raise SystemExit(f"Expected ready enablement metadata: {metadata}")
                if "search mailbox metadata" in case:
                    assert_contains(
                        result.response,
                        [
                            "Enablement verdict: READY_FOR_APPROVAL_GATED_ENABLEMENT_REVIEW",
                            "Suggested risk: PERSONAL_DATA",
                            "Approval required for real use: required before any real personal-data read or side effect",
                            "disabled adapter acceptance: ADAPTER_ACCEPTANCE_PASSED; 4/4 disabled adapter cases passed",
                            "metadata row contract: ready",
                            "metadata row schema: id, timestamp, label, source",
                            "metadata row limit: 2",
                            "disabled adapter acceptance proof for metadata-only routes",
                            "missing blockers: none",
                            "natural-language auto-routing: disabled",
                        ],
                        case,
                    )
                    if (
                        metadata.get("enablement_verdict") != "READY_FOR_APPROVAL_GATED_ENABLEMENT_REVIEW"
                        or metadata.get("missing_fields")
                        or metadata.get("suggested_risk") != "PERSONAL_DATA"
                        or not metadata.get("adapter_acceptance_required")
                        or metadata.get("adapter_acceptance_state") != "ADAPTER_ACCEPTANCE_PASSED"
                        or metadata.get("adapter_acceptance_case_count") != 4
                        or not metadata.get("adapter_acceptance_passed")
                        or metadata.get("metadata_row_contract_required") is not True
                        or metadata.get("metadata_row_contract_ready") is not True
                        or metadata.get("sample_row_schema") != ["id", "timestamp", "label", "source"]
                        or metadata.get("sample_row_limit") != 2
                        or metadata.get("natural_language_routing_enabled")
                        or not metadata.get("requires_approval")
                        or not metadata.get("implementation_review_allowed")
                    ):
                        raise SystemExit(f"Expected metadata enablement with adapter acceptance proof: {metadata}")
                if "logged-in page" in case:
                    assert_contains(result.response, ["Enablement verdict: ENABLEMENT_BLOCKED", "missing blockers:", "target/source", "smoke test plan", "audit trail plan", "acceptance-gate evidence"], case)
                    if metadata.get("enablement_verdict") != "ENABLEMENT_BLOCKED" or not metadata.get("missing_fields") or metadata.get("implementation_review_allowed"):
                        raise SystemExit(f"Expected blocked enablement metadata: {metadata}")
            elif "rehearsal receipt" in case:
                assert_contains(
                    result.response,
                    [
                        "Jarvis personal integration rehearsal receipt",
                        "This is read-only",
                        "Rehearsal state",
                        "Exact future execution envelope",
                        "Rehearsed lifecycle",
                        "Proof carried into implementation",
                        "adapter state: disabled",
                        "Known connector boundaries",
                        "Stop conditions",
                    ],
                    case,
                )
                metadata = result.tool_results[0].metadata
                assert_read_only_metadata(metadata, "Integration rehearsal receipt")
                handoff = metadata.get("rehearsal_handoff")
                if not isinstance(handoff, dict):
                    raise SystemExit(f"Expected structured rehearsal handoff: {metadata}")
                handoff_scope = handoff.get("scope") or {}
                handoff_proof = handoff.get("proof") or {}
                handoff_contract = handoff.get("metadata_row_contract") or {}
                handoff_risk = handoff.get("risk") or {}
                handoff_lifecycle = handoff.get("lifecycle") or {}
                handoff_boundaries = handoff.get("boundaries") or {}
                handoff_next_commands = handoff.get("next_commands") or {}
                if (
                    metadata.get("rehearsal_handoff_ready") != handoff.get("handoff_ready")
                    or handoff.get("rehearsal_handoff_ready") != handoff.get("handoff_ready")
                    or handoff.get("handoff_ready") is not True
                    or metadata.get("rehearsal_ready_for_operator") != handoff.get("ready_for_operator")
                    or handoff.get("ready_for_operator") is not True
                    or metadata.get("rehearsal_state_changed") != handoff.get("state_changed")
                    or handoff.get("state_changed") is not False
                    or metadata.get("rehearsal_changed") != handoff.get("changed")
                    or handoff.get("changed") != []
                    or metadata.get("rehearsal_content_in_handoff") != handoff.get("content_in_handoff")
                    or handoff.get("content_in_handoff") is not True
                    or metadata.get("rehearsal_authorizes_execution") != handoff.get("authorizes_execution")
                    or handoff.get("authorizes_execution") is not False
                    or metadata.get("rehearsal_authorizes_completion_claim") != handoff.get("authorizes_completion_claim")
                    or handoff.get("authorizes_completion_claim") is not False
                    or metadata.get("rehearsal_approval_granted") != handoff.get("approval_granted")
                    or handoff.get("approval_granted") is not False
                    or metadata.get("rehearsal_boundaries") != handoff_boundaries
                    or metadata.get("rehearsal_next_safe_commands") != handoff.get("next_safe_commands")
                    or list(handoff_next_commands.values()) != handoff.get("next_safe_commands")
                    or len(handoff_next_commands) < 3
                    or handoff.get("connector") != metadata.get("connector")
                    or handoff.get("requested_connector") != metadata.get("requested_connector")
                    or handoff.get("action") != metadata.get("action")
                    or handoff.get("tool_name") != metadata.get("tool_name")
                    or handoff.get("rehearsal_state") != metadata.get("rehearsal_state")
                    or handoff.get("rehearsal_ready") != metadata.get("rehearsal_ready")
                    or handoff.get("adapter_state") != metadata.get("adapter_state")
                    or handoff.get("simulated_result") != metadata.get("simulated_result")
                    or handoff.get("missing_fields") != metadata.get("missing_fields")
                    or handoff_scope.get("target") != metadata.get("target")
                    or handoff_scope.get("time_range") != metadata.get("time_range")
                    or handoff_scope.get("data_level") != metadata.get("data_level")
                    or handoff_scope.get("verification") != metadata.get("verification")
                    or handoff_scope.get("rollback") != metadata.get("rollback")
                    or handoff_proof.get("tests") != metadata.get("tests")
                    or handoff_proof.get("audit") != metadata.get("audit")
                    or handoff_proof.get("acceptance") != metadata.get("acceptance")
                    or handoff_contract.get("required") != metadata.get("metadata_row_contract_required")
                    or handoff_contract.get("ready") != metadata.get("metadata_row_contract_ready")
                    or handoff_contract.get("sample_row_schema") != metadata.get("sample_row_schema")
                    or handoff_contract.get("sample_row_limit") != metadata.get("sample_row_limit")
                    or handoff_contract.get("blocked_payload_fields") != metadata.get("blocked_payload_fields")
                    or handoff_risk.get("suggested_risk") != metadata.get("suggested_risk")
                    or handoff_risk.get("approval_required") != metadata.get("approval_required")
                    or handoff_risk.get("requires_approval") != metadata.get("requires_approval")
                    or handoff_risk.get("rollback_required") != metadata.get("rollback_required")
                    or not handoff_lifecycle.get("preflight")
                    or not handoff_lifecycle.get("gate")
                    or not handoff_lifecycle.get("adapter_boundary")
                    or handoff.get("next_step_requires_proof_bundle") is not True
                    or handoff.get("next_route_lock_requires_fresh_review") is not True
                    or handoff_boundaries.get("natural_language_routing_enabled")
                    or handoff_boundaries.get("adapter_default_state") != "disabled"
                    or not handoff_boundaries.get("real_adapter_call_skipped")
                    or handoff_boundaries.get("calls_external_service")
                    or handoff_boundaries.get("reads_personal_data")
                    or handoff_boundaries.get("executes_side_effect")
                    or handoff_boundaries.get("writes_memory")
                    or handoff_boundaries.get("authorizes_account_access")
                    or handoff_boundaries.get("authorizes_route_unlock")
                    or handoff_boundaries.get("authorizes_natural_language_routing")
                    or handoff_boundaries.get("authorizes_personal_data_read")
                    or handoff_boundaries.get("authorizes_side_effect")
                    or handoff_boundaries.get("authorizes_approval")
                    or handoff_boundaries.get("authorizes_model_call")
                    or handoff_boundaries.get("authorizes_tool_execution")
                    or handoff_boundaries.get("authorizes_external_service")
                    or handoff_boundaries.get("authorizes_execution")
                    or handoff_boundaries.get("authorizes_completion_claim")
                    or handoff_boundaries.get("approval_granted")
                    or handoff_boundaries.get("reusable_for_other_scope")
                ):
                    raise SystemExit(f"Rehearsal handoff diverged from flat metadata or safety contract: {metadata}")
                if "send draft reply" in case:
                    assert_contains(
                        result.response,
                        [
                            "Rehearsal state: REHEARSAL_READY_APPROVAL_GATED",
                            "Suggested risk: EXTERNAL_SIDE_EFFECT",
                            "Approval checkpoint: required before adapter call in any real run",
                            "smoke tests: blocked send smoke",
                            "audit trail: tool run receipt",
                            "acceptance gate: gate passed with receipts",
                            "missing blockers: none",
                            "simulated result: no connector call made",
                            "metadata row contract: not required for this rehearsal",
                        ],
                        case,
                    )
                    if (
                        metadata.get("rehearsal_state") != "REHEARSAL_READY_APPROVAL_GATED"
                        or metadata.get("missing_fields")
                        or metadata.get("suggested_risk") != "EXTERNAL_SIDE_EFFECT"
                        or metadata.get("adapter_state") != "disabled"
                        or metadata.get("natural_language_routing_enabled")
                        or not metadata.get("rollback_required")
                        or not metadata.get("rehearsal_ready")
                        or metadata.get("metadata_row_contract_required")
                        or metadata.get("metadata_row_contract_ready")
                    ):
                        raise SystemExit(f"Expected ready rehearsal metadata: {metadata}")
                if "search mailbox metadata" in case:
                    assert_contains(
                        result.response,
                        [
                            "Rehearsal state: REHEARSAL_READY_APPROVAL_GATED",
                            "metadata row contract: ready",
                            "metadata row schema: id, timestamp, label, source",
                            "metadata row limit: 2",
                            "blocked payload fields: body, content, notes, attachments, tokens, cookies, contact_cards, authenticated_page_content",
                            "missing blockers: none",
                        ],
                        case,
                    )
                    if (
                        metadata.get("rehearsal_state") != "REHEARSAL_READY_APPROVAL_GATED"
                        or metadata.get("missing_fields")
                        or metadata.get("adapter_state") != "disabled"
                        or metadata.get("metadata_row_contract_required") is not True
                        or metadata.get("metadata_row_contract_ready") is not True
                        or metadata.get("sample_row_schema") != ["id", "timestamp", "label", "source"]
                        or metadata.get("sample_row_limit") != 2
                        or metadata.get("natural_language_routing_enabled")
                        or not metadata.get("rehearsal_ready")
                    ):
                        raise SystemExit(f"Expected metadata rehearsal row-contract metadata: {metadata}")
                if "logged-in page" in case:
                    assert_contains(result.response, ["Rehearsal state: REHEARSAL_BLOCKED", "missing blockers:", "target/source", "smoke test plan", "audit trail plan", "acceptance-gate evidence"], case)
                    if (
                        metadata.get("rehearsal_state") != "REHEARSAL_BLOCKED"
                        or not metadata.get("missing_fields")
                        or metadata.get("rehearsal_ready")
                        or metadata.get("rehearsal_handoff_ready") is not True
                        or metadata.get("rehearsal_ready_for_operator") is not True
                    ):
                        raise SystemExit(f"Expected blocked rehearsal metadata: {metadata}")
            elif "route lock" in case:
                assert_contains(
                    result.response,
                    [
                        "Jarvis personal integration route lock",
                        "This is read-only",
                        "Route lock state",
                        "Route posture",
                        "natural-language auto-routing: disabled",
                        "Route lock rules",
                        "Explicit route review contract",
                        "authorizes account access: no",
                        "authorizes route unlock: no",
                        "authorizes natural-language routing: no",
                        "authorizes personal-data read: no",
                        "authorizes side effect: no",
                        "reusable for other scope: no",
                        "Route lock token boundary",
                        "route-lock tokens are proof-only",
                        "Route proof queue",
                        "next route required:",
                        "route proof queue count:",
                        "Known connector boundaries",
                        "Stop conditions",
                    ],
                    case,
                )
                metadata = result.tool_results[0].metadata
                assert_read_only_metadata(metadata, "Integration route lock")
                route_queue = metadata.get("route_proof_queue") or []
                if not route_queue or metadata.get("route_proof_queue_count") != len(route_queue):
                    raise SystemExit(f"Integration route lock missed route proof queue metadata: {metadata}")
                if metadata.get("integration_proof_queue") != route_queue:
                    raise SystemExit(f"Integration route lock proof queue alias diverged: {metadata}")
                if metadata.get("integration_proof_queue_count") != len(route_queue):
                    raise SystemExit(f"Integration route lock proof queue count alias diverged: {metadata}")
                if metadata.get("next_required_command") != metadata.get("next_route_proof_command"):
                    raise SystemExit(f"Integration route lock next required alias diverged: {metadata}")
                if metadata.get("next_route_required_command") != metadata.get("next_route_proof_command"):
                    raise SystemExit(f"Integration route lock next route required alias diverged: {metadata}")
                if metadata.get("integration_next_required_command") != metadata.get("next_required_command"):
                    raise SystemExit(f"Integration route lock next required integration alias diverged: {metadata}")
                if metadata.get("integration_next_proof_command") != metadata.get("next_route_proof_command"):
                    raise SystemExit(f"Integration route lock next proof alias diverged: {metadata}")
                if "next route proof:" in result.response:
                    raise SystemExit("Integration route lock should render next route required, not next route proof.")
                if metadata.get("natural_language_routing_enabled"):
                    raise SystemExit(f"Integration route lock should never enable natural-language routing directly: {metadata}")
                if metadata.get("explicit_route_review_required") is not True:
                    raise SystemExit(f"Integration route lock should require explicit route review: {metadata}")
                route_review_rows = metadata.get("route_review_contract_rows") or []
                implementation_review_receipt_rows = metadata.get("implementation_review_receipt_contract_rows") or []
                expected_route_review_items = {
                    "implementation_review_receipt",
                    "exact_scope_hash",
                    "status_api_smoke_evidence",
                    "metadata_row_contract",
                    "explicit_route_review_required",
                    "approval_chain_boundary",
                }
                if (
                    len(route_review_rows) != 6
                    or metadata.get("route_review_contract_row_count") != len(route_review_rows)
                    or metadata.get("route_review_contract_ready") is not True
                    or expected_route_review_items - {row.get("item") for row in route_review_rows}
                    or metadata.get("route_review_authorizes_account_access")
                    or metadata.get("route_review_authorizes_route_unlock")
                    or metadata.get("route_review_authorizes_natural_language_routing")
                    or metadata.get("route_review_authorizes_personal_data_read")
                    or metadata.get("route_review_authorizes_side_effect")
                    or metadata.get("route_review_reusable_for_other_scope")
                    or any(
                        row.get("authorizes_account_access")
                        or row.get("authorizes_route_unlock")
                        or row.get("authorizes_natural_language_routing")
                        or row.get("authorizes_personal_data_read")
                        or row.get("authorizes_side_effect")
                        or row.get("reusable_for_other_scope")
                        for row in route_review_rows
                    )
                ):
                    raise SystemExit(f"Integration route lock route-review contract unsafe: {metadata}")
                if not _route_review_contract_ready(
                    route_review_rows,
                    route_unlock_candidate=bool(metadata.get("route_unlock_candidate")),
                    scope_hash=str(metadata.get("integration_scope_hash") or ""),
                    review_receipt_id=str(metadata.get("review_receipt_id") or ""),
                    route_lock_state=str(metadata.get("route_lock_state") or ""),
                ):
                    raise SystemExit(f"Integration route lock route-review contract failed production validator: {metadata}")
                tampered_route_review_contract_rows = [dict(row) for row in route_review_rows]
                tampered_route_review_contract_rows[0]["authorizes_route_unlock"] = True
                if _route_review_contract_ready(
                    tampered_route_review_contract_rows,
                    route_unlock_candidate=bool(metadata.get("route_unlock_candidate")),
                    scope_hash=str(metadata.get("integration_scope_hash") or ""),
                    review_receipt_id=str(metadata.get("review_receipt_id") or ""),
                    route_lock_state=str(metadata.get("route_lock_state") or ""),
                ):
                    raise SystemExit(f"Integration route lock route-review validator should reject authority tampering: {metadata}")
                for expected_summary in [
                    "explicit-review-only",
                    "account-access-not-authorized",
                    "route-unlock-not-authorized",
                    "natural-language-routing-not-authorized",
                    "approval-chain-still-required",
                    "fresh-scope-review-required",
                ]:
                    if expected_summary not in metadata.get("route_review_contract_summary", []):
                        raise SystemExit(f"Integration route lock missed route-review summary {expected_summary}: {metadata}")
                if (
                    len(implementation_review_receipt_rows) != 5
                    or metadata.get("implementation_review_receipt_contract_row_count") != len(implementation_review_receipt_rows)
                    or metadata.get("route_lock_token_binds_implementation_review_receipt") is not True
                    or any(row.get("authorizes_account_access") or row.get("authorizes_route_unlock") or row.get("reusable_for_other_scope") for row in implementation_review_receipt_rows)
                ):
                    raise SystemExit(f"Integration route lock missed safe upstream implementation receipt binding: {metadata}")
                if not _implementation_review_receipt_contract_ready(
                    implementation_review_receipt_rows,
                    expected_ready=bool(metadata.get("implementation_review_ready")),
                    scope_hash=str(metadata.get("integration_scope_hash") or ""),
                    review_receipt_id=str(metadata.get("review_receipt_id") or ""),
                ):
                    raise SystemExit(f"Integration route lock upstream implementation receipt failed production validator: {metadata}")
                route_lock_token = str(metadata.get("route_lock_token_sha256") or "")
                route_token_rows = metadata.get("route_lock_token_boundary_rows") or []
                expected_route_token_items = {
                    "route_lock_token",
                    "explicit_route_review_only",
                    "natural_language_routing_disabled",
                    "next_connector_action_approval_boundary",
                }
                if (
                    len(route_lock_token) != 64
                    or any(char not in "0123456789abcdefABCDEF" for char in route_lock_token)
                    or metadata.get("route_lock_token_present") is not True
                    or metadata.get("route_lock_token_boundary_row_count") != len(route_token_rows)
                    or len(route_token_rows) != 4
                    or metadata.get("route_lock_token_boundary_ready") is not True
                    or expected_route_token_items - {row.get("item") for row in route_token_rows}
                    or metadata.get("route_lock_token_authorizes_account_access")
                    or metadata.get("route_lock_token_authorizes_route_unlock")
                    or metadata.get("route_lock_token_authorizes_natural_language_routing")
                    or metadata.get("route_lock_token_authorizes_personal_data_read")
                    or metadata.get("route_lock_token_authorizes_side_effect")
                    or metadata.get("route_lock_token_authorizes_approval")
                    or metadata.get("route_lock_token_authorizes_model_call")
                    or metadata.get("route_lock_token_authorizes_tool_execution")
                    or metadata.get("route_lock_token_authorizes_external_service")
                    or metadata.get("route_lock_token_reusable_for_other_scope")
                    or metadata.get("route_lock_token_reusable_for_next_route_review")
                    or metadata.get("next_integration_route_requires_fresh_route_lock_review") is not True
                    or any(
                        row.get("token_sha256") != route_lock_token
                        or row.get("authorizes_account_access")
                        or row.get("authorizes_route_unlock")
                        or row.get("authorizes_natural_language_routing")
                        or row.get("authorizes_personal_data_read")
                        or row.get("authorizes_side_effect")
                        or row.get("authorizes_approval")
                        or row.get("authorizes_model_call")
                        or row.get("authorizes_tool_execution")
                        or row.get("authorizes_external_service")
                        or row.get("reusable_for_other_scope")
                        or row.get("reusable_for_next_route_review")
                        for row in route_token_rows
                    )
                ):
                    raise SystemExit(f"Integration route lock token boundary unsafe: {metadata}")
                if not _route_lock_token_boundary_ready(route_lock_token, route_token_rows):
                    raise SystemExit(f"Integration route lock token boundary failed production validator: {metadata}")
                tampered_route_token_rows = [dict(row) for row in route_token_rows]
                tampered_route_token_rows[0]["authorizes_personal_data_read"] = True
                if _route_lock_token_boundary_ready(route_lock_token, tampered_route_token_rows):
                    raise SystemExit(f"Integration route lock token boundary validator should reject authority tampering: {metadata}")
                tampered_route_token_status_rows = [dict(row) for row in route_token_rows]
                tampered_route_token_status_rows[2]["status"] = "natural_language_routing_enabled"
                if _route_lock_token_boundary_ready(route_lock_token, tampered_route_token_status_rows):
                    raise SystemExit(f"Integration route lock token boundary validator should reject status tampering: {metadata}")
                route_lock_handoff = metadata.get("route_lock_handoff") or {}
                handoff_token = route_lock_handoff.get("route_lock_token") or {}
                handoff_contract = route_lock_handoff.get("route_review_contract") or {}
                handoff_receipt = route_lock_handoff.get("implementation_review_receipt") or {}
                handoff_queue = route_lock_handoff.get("proof_queue") or {}
                handoff_authorizes = route_lock_handoff.get("authorizes") or {}
                handoff_boundaries = route_lock_handoff.get("boundaries") or {}
                handoff_next_commands = route_lock_handoff.get("next_commands") or {}
                if (
                    metadata.get("route_lock_handoff_ready") != route_lock_handoff.get("handoff_ready")
                    or route_lock_handoff.get("route_lock_handoff_ready") != route_lock_handoff.get("handoff_ready")
                    or route_lock_handoff.get("handoff_ready") is not True
                    or metadata.get("route_lock_ready_for_operator") != route_lock_handoff.get("ready_for_operator")
                    or route_lock_handoff.get("ready_for_operator") is not True
                    or metadata.get("route_lock_state_changed") != route_lock_handoff.get("state_changed")
                    or route_lock_handoff.get("state_changed") is not False
                    or metadata.get("route_lock_changed") != route_lock_handoff.get("changed")
                    or route_lock_handoff.get("changed") != []
                    or metadata.get("route_lock_content_in_handoff") != route_lock_handoff.get("content_in_handoff")
                    or route_lock_handoff.get("content_in_handoff") is not True
                    or metadata.get("route_lock_authorizes_execution") != route_lock_handoff.get("authorizes_execution")
                    or route_lock_handoff.get("authorizes_execution") is not False
                    or metadata.get("route_lock_authorizes_completion_claim") != route_lock_handoff.get("authorizes_completion_claim")
                    or route_lock_handoff.get("authorizes_completion_claim") is not False
                    or metadata.get("route_lock_approval_granted") != route_lock_handoff.get("approval_granted")
                    or route_lock_handoff.get("approval_granted") is not False
                    or metadata.get("route_lock_boundaries") != handoff_boundaries
                    or metadata.get("route_lock_next_safe_commands") != route_lock_handoff.get("next_safe_commands")
                    or list(handoff_next_commands.values()) != route_lock_handoff.get("next_safe_commands")
                    or len(handoff_next_commands) != 2
                    or route_lock_handoff.get("connector") != metadata.get("connector")
                    or route_lock_handoff.get("action") != metadata.get("action")
                    or route_lock_handoff.get("route_lock_state") != metadata.get("route_lock_state")
                    or route_lock_handoff.get("route_unlock_candidate") != metadata.get("route_unlock_candidate")
                    or route_lock_handoff.get("explicit_route_review_required") is not True
                    or route_lock_handoff.get("natural_language_routing_enabled")
                    or route_lock_handoff.get("implementation_review_ready") != metadata.get("implementation_review_ready")
                    or route_lock_handoff.get("status_evidence_present") != metadata.get("status_evidence_present")
                    or route_lock_handoff.get("metadata_row_contract_ready") != metadata.get("metadata_row_contract_ready")
                    or route_lock_handoff.get("review_receipt_id") != metadata.get("review_receipt_id")
                    or route_lock_handoff.get("integration_scope_hash") != metadata.get("integration_scope_hash")
                    or route_lock_handoff.get("expected_scope_hash") != metadata.get("expected_scope_hash")
                    or route_lock_handoff.get("expected_scope_hash_required") != metadata.get("expected_scope_hash_required")
                    or route_lock_handoff.get("expected_scope_hash_valid") != metadata.get("expected_scope_hash_valid")
                    or route_lock_handoff.get("scope_hash_matches_expected") != metadata.get("scope_hash_matches_expected")
                    or route_lock_handoff.get("missing_fields") != metadata.get("missing_fields")
                    or handoff_contract.get("ready") is not True
                    or handoff_contract.get("row_count") != len(route_review_rows)
                    or handoff_contract.get("rows") != route_review_rows
                    or handoff_contract.get("summary") != metadata.get("route_review_contract_summary")
                    or handoff_receipt.get("row_count") != len(implementation_review_receipt_rows)
                    or handoff_receipt.get("rows") != implementation_review_receipt_rows
                    or handoff_token.get("sha256") != route_lock_token
                    or handoff_token.get("present") is not True
                    or handoff_token.get("boundary_ready") is not True
                    or handoff_token.get("boundary_row_count") != len(route_token_rows)
                    or handoff_token.get("boundary_rows") != route_token_rows
                    or handoff_queue.get("items") != route_queue
                    or handoff_queue.get("count") != len(route_queue)
                    or handoff_queue.get("next") != metadata.get("next_route_proof_command")
                    or handoff_queue.get("next_required") != metadata.get("next_required_command")
                    or handoff_queue.get("next_proof") != metadata.get("next_route_proof_command")
                    or route_lock_handoff.get("commands", {}).get("implementation_review") != metadata.get("implementation_review_command")
                    or route_lock_handoff.get("commands", {}).get("route_lock") != metadata.get("route_lock_command")
                    or handoff_next_commands != route_lock_handoff.get("commands", {})
                    or route_lock_handoff.get("fresh_review_required_for_next_route") is not True
                    or any(handoff_authorizes.values())
                    or any(handoff_boundaries.values())
                    or handoff_contract.get("authorizes_account_access")
                    or handoff_contract.get("authorizes_route_unlock")
                    or handoff_contract.get("authorizes_natural_language_routing")
                    or handoff_contract.get("authorizes_personal_data_read")
                    or handoff_contract.get("authorizes_side_effect")
                    or handoff_contract.get("reusable_for_other_scope")
                    or handoff_receipt.get("authorizes_account_access")
                    or handoff_receipt.get("authorizes_route_unlock")
                    or handoff_receipt.get("reusable_for_other_scope")
                    or handoff_token.get("authorizes_account_access")
                    or handoff_token.get("authorizes_route_unlock")
                    or handoff_token.get("authorizes_natural_language_routing")
                    or handoff_token.get("authorizes_personal_data_read")
                    or handoff_token.get("authorizes_side_effect")
                    or handoff_token.get("authorizes_approval")
                    or handoff_token.get("authorizes_model_call")
                    or handoff_token.get("authorizes_tool_execution")
                    or handoff_token.get("authorizes_external_service")
                    or handoff_token.get("reusable_for_other_scope")
                    or handoff_token.get("reusable_for_next_route_review")
                ):
                    raise SystemExit(f"Integration route lock handoff diverged from safety contract: {metadata}")
                recomputed_route_lock_token = _route_lock_token_sha256(
                    scope_hash=str(metadata.get("integration_scope_hash") or ""),
                    review_receipt_id=str(metadata.get("review_receipt_id") or ""),
                    route_lock_state=str(metadata.get("route_lock_state") or ""),
                    expected_scope_hash=str(metadata.get("expected_scope_hash") or ""),
                    implementation_review_receipt_contract_rows=implementation_review_receipt_rows,
                    route_review_contract_rows=route_review_rows,
                )
                if recomputed_route_lock_token != route_lock_token:
                    raise SystemExit(f"Integration route lock token should be reproducible: {metadata}")
                malformed_closed_receipt_rows = [dict(row) for row in implementation_review_receipt_rows]
                malformed_closed_receipt_rows[0]["authorizes_account_access"] = "false"
                malformed_closed_route_rows = [dict(row) for row in route_review_rows]
                malformed_closed_route_rows[0]["authorizes_personal_data_read"] = "false"
                malformed_closed_route_lock_token = _route_lock_token_sha256(
                    scope_hash=str(metadata.get("integration_scope_hash") or ""),
                    review_receipt_id=str(metadata.get("review_receipt_id") or ""),
                    route_lock_state=str(metadata.get("route_lock_state") or ""),
                    expected_scope_hash=str(metadata.get("expected_scope_hash") or ""),
                    implementation_review_receipt_contract_rows=malformed_closed_receipt_rows,
                    route_review_contract_rows=malformed_closed_route_rows,
                )
                if malformed_closed_route_lock_token != route_lock_token:
                    raise SystemExit(f"Integration route lock token should default malformed closed flags to false: {metadata}")
                tampered_implementation_review_receipt_rows = [dict(row) for row in implementation_review_receipt_rows]
                tampered_implementation_review_receipt_rows[0]["authorizes_account_access"] = True
                tampered_receipt_route_lock_token = _route_lock_token_sha256(
                    scope_hash=str(metadata.get("integration_scope_hash") or ""),
                    review_receipt_id=str(metadata.get("review_receipt_id") or ""),
                    route_lock_state=str(metadata.get("route_lock_state") or ""),
                    expected_scope_hash=str(metadata.get("expected_scope_hash") or ""),
                    implementation_review_receipt_contract_rows=tampered_implementation_review_receipt_rows,
                    route_review_contract_rows=route_review_rows,
                )
                if tampered_receipt_route_lock_token == route_lock_token:
                    raise SystemExit(f"Integration route lock token should bind upstream implementation receipt authority flags: {metadata}")
                tampered_route_review_rows = [dict(row) for row in route_review_rows]
                tampered_route_review_rows[0]["authorizes_personal_data_read"] = True
                tampered_route_lock_token = _route_lock_token_sha256(
                    scope_hash=str(metadata.get("integration_scope_hash") or ""),
                    review_receipt_id=str(metadata.get("review_receipt_id") or ""),
                    route_lock_state=str(metadata.get("route_lock_state") or ""),
                    expected_scope_hash=str(metadata.get("expected_scope_hash") or ""),
                    implementation_review_receipt_contract_rows=implementation_review_receipt_rows,
                    route_review_contract_rows=tampered_route_review_rows,
                )
                if tampered_route_lock_token == route_lock_token:
                    raise SystemExit(f"Integration route lock token should bind route-review authority flags: {metadata}")
                for command in [
                    "integration implementation review:",
                    "integration adapter manifest:",
                    "integration adapter acceptance:",
                    "execution health report",
                    "integration route lock:",
                ]:
                    if not any(str(item).startswith(command) for item in route_queue):
                        raise SystemExit(f"Integration route lock proof queue missed {command}: {metadata}")
                if "search mailbox metadata" in case and "status status api smoke passed" in case and "expected scope hash" not in case:
                    assert_contains(
                        result.response,
                        [
                            "Route lock state: ROUTE_LOCK_READY_FOR_EXPLICIT_REVIEW",
                            "implementation review ready: yes",
                            "status/API smoke evidence: present",
                            "metadata row contract: ready",
                            "route unlock candidate: yes, explicit review still required",
                            "missing blockers: none",
                            "next route required: `execution health report`",
                        ],
                        case,
                    )
                    if (
                        metadata.get("route_lock_state") != "ROUTE_LOCK_READY_FOR_EXPLICIT_REVIEW"
                        or metadata.get("route_unlock_candidate") is not True
                        or metadata.get("implementation_review_ready") is not True
                        or metadata.get("status_evidence_present") is not True
                        or metadata.get("metadata_row_contract_ready") is not True
                        or metadata.get("next_route_proof_command") != "execution health report"
                        or not metadata.get("review_receipt_id")
                        or len(str(metadata.get("integration_scope_hash") or "")) != 16
                    ):
                        raise SystemExit(f"Expected route lock ready metadata: {metadata}")
                if "search mailbox metadata" in case and "expected scope hash deadbeefdeadbeef" in case:
                    assert_contains(
                        result.response,
                        [
                            "Route lock state: ROUTE_LOCK_HELD",
                            "status/API smoke evidence: present",
                            "expected scope hash: `deadbeefdeadbeef`",
                            "expected scope hash valid: yes",
                            "expected scope hash match: no",
                            "route unlock candidate: no",
                            "expected scope hash match",
                        ],
                        case,
                    )
                    if (
                        metadata.get("route_lock_state") != "ROUTE_LOCK_HELD"
                        or metadata.get("route_unlock_candidate")
                        or metadata.get("scope_hash_matches_expected")
                        or metadata.get("expected_scope_hash_required") is not True
                        or metadata.get("expected_scope_hash_valid") is not True
                        or "expected scope hash match" not in metadata.get("missing_fields", [])
                    ):
                        raise SystemExit(f"Expected route lock held on expected hash mismatch: {metadata}")
                if "search mailbox metadata" in case and "expected scope hash not-a-scope-hash" in case:
                    assert_contains(
                        result.response,
                        [
                            "Route lock state: ROUTE_LOCK_HELD",
                            "status/API smoke evidence: present",
                            "expected scope hash: `not-a-scope-hash`",
                            "expected scope hash valid: no",
                            "expected scope hash match: no",
                            "route unlock candidate: no",
                            "valid expected scope hash",
                        ],
                        case,
                    )
                    if (
                        metadata.get("route_lock_state") != "ROUTE_LOCK_HELD"
                        or metadata.get("route_unlock_candidate")
                        or metadata.get("scope_hash_matches_expected")
                        or metadata.get("expected_scope_hash_required") is not True
                        or metadata.get("expected_scope_hash_valid") is not False
                        or "valid expected scope hash" not in metadata.get("missing_fields", [])
                        or "expected scope hash match" not in metadata.get("missing_fields", [])
                    ):
                        raise SystemExit(f"Expected route lock held on invalid expected hash: {metadata}")
                if "search mailbox metadata" in case and "status status api smoke passed" not in case:
                    assert_contains(
                        result.response,
                        [
                            "Route lock state: ROUTE_LOCK_HELD",
                            "implementation review ready: no",
                            "status/API smoke evidence: missing",
                            "route unlock candidate: no",
                            "status/API smoke evidence",
                        ],
                        case,
                    )
                    if (
                        metadata.get("route_lock_state") != "ROUTE_LOCK_HELD"
                        or metadata.get("route_unlock_candidate")
                        or metadata.get("status_evidence_present")
                        or "status/API smoke evidence" not in metadata.get("missing_fields", [])
                    ):
                        raise SystemExit(f"Expected route lock held without status metadata: {metadata}")
                if "logged-in page" in case:
                    assert_contains(result.response, ["Route lock state: ROUTE_LOCK_HELD", "missing blockers:", "implementation review ready receipt"], case)
                    if metadata.get("route_lock_state") != "ROUTE_LOCK_HELD" or metadata.get("route_unlock_candidate"):
                        raise SystemExit(f"Expected blocked route lock metadata: {metadata}")
            elif "implementation review" in case:
                assert_contains(
                    result.response,
                    [
                        "Jarvis personal integration implementation review",
                        "This is read-only",
                        "Review state",
                        "Exact implementation review scope",
                        "Review checks",
                        "proof bundle",
                        "metadata row contract",
                        "preflight contract",
                        "implementation spec",
                        "status visibility",
                        "Implementation invariants before code review",
                        "receipt contract rows:",
                        "receipt contract summary:",
                        "Implementation proof queue",
                        "proof queue count",
                        "proof queue:",
                        "Stop conditions",
                    ],
                    case,
                )
                metadata = result.tool_results[0].metadata
                assert_read_only_metadata(metadata, "Integration implementation review")
                review_proof_queue = metadata.get("proof_queue") or []
                if not review_proof_queue or metadata.get("proof_queue_count") != len(review_proof_queue):
                    raise SystemExit(f"Integration implementation review missed proof queue metadata: {metadata}")
                if metadata.get("next_proof_command") not in review_proof_queue and metadata.get("next_proof_command") != "execution health report":
                    raise SystemExit(f"Integration implementation review missed next proof command: {metadata}")
                if metadata.get("next_required_command") != metadata.get("next_proof_command"):
                    raise SystemExit(f"Integration implementation review next required alias diverged: {metadata}")
                if metadata.get("integration_proof_queue") != review_proof_queue:
                    raise SystemExit(f"Integration implementation review proof queue alias diverged: {metadata}")
                if metadata.get("integration_proof_queue_count") != len(review_proof_queue):
                    raise SystemExit(f"Integration implementation review proof queue count alias diverged: {metadata}")
                if metadata.get("integration_next_required_command") != metadata.get("next_required_command"):
                    raise SystemExit(f"Integration implementation review next required integration alias diverged: {metadata}")
                if metadata.get("integration_next_proof_command") != metadata.get("next_proof_command"):
                    raise SystemExit(f"Integration implementation review next proof alias diverged: {metadata}")
                receipt_rows = metadata.get("implementation_review_receipt_contract_rows") or []
                if (
                    metadata.get("implementation_review_receipt_contract_row_count") != len(receipt_rows)
                    or len(receipt_rows) != 5
                    or metadata.get("implementation_review_receipt_authorizes_account_access")
                    or metadata.get("implementation_review_receipt_authorizes_route_unlock")
                    or metadata.get("implementation_review_receipt_reusable_for_other_scope")
                    or any(row.get("authorizes_account_access") or row.get("authorizes_route_unlock") or row.get("reusable_for_other_scope") for row in receipt_rows)
                ):
                    raise SystemExit(f"Integration implementation review receipt contract unsafe: {metadata}")
                implementation_handoff = metadata.get("implementation_review_handoff") or {}
                handoff_receipt = implementation_handoff.get("review_receipt") or {}
                handoff_queue = implementation_handoff.get("proof_queue") or {}
                handoff_commands = implementation_handoff.get("commands") or {}
                handoff_states = implementation_handoff.get("source_states") or {}
                handoff_risk = implementation_handoff.get("risk") or {}
                handoff_boundaries = implementation_handoff.get("boundaries") or {}
                handoff_next_commands = implementation_handoff.get("next_commands") or {}
                expected_review_ready = metadata.get("review_state") == "IMPLEMENTATION_REVIEW_READY"
                if not _implementation_review_receipt_contract_ready(
                    receipt_rows,
                    expected_ready=expected_review_ready,
                    scope_hash=str(metadata.get("integration_scope_hash") or ""),
                    review_receipt_id=str(metadata.get("review_receipt_id") or ""),
                ):
                    raise SystemExit(f"Integration implementation review receipt failed production validator: {metadata}")
                tampered_receipt_rows = [dict(row) for row in receipt_rows]
                tampered_receipt_rows[0]["authorizes_account_access"] = True
                if _implementation_review_receipt_contract_ready(
                    tampered_receipt_rows,
                    expected_ready=expected_review_ready,
                    scope_hash=str(metadata.get("integration_scope_hash") or ""),
                    review_receipt_id=str(metadata.get("review_receipt_id") or ""),
                ):
                    raise SystemExit(f"Integration implementation review receipt validator should reject authority tampering: {metadata}")
                if (
                    metadata.get("implementation_review_handoff_ready") != implementation_handoff.get("handoff_ready")
                    or implementation_handoff.get("implementation_review_handoff_ready") != implementation_handoff.get("handoff_ready")
                    or implementation_handoff.get("handoff_ready") is not True
                    or metadata.get("implementation_review_ready_for_operator") != implementation_handoff.get("ready_for_operator")
                    or implementation_handoff.get("ready_for_operator") is not True
                    or metadata.get("implementation_review_state_changed") != implementation_handoff.get("state_changed")
                    or implementation_handoff.get("state_changed") is not False
                    or metadata.get("implementation_review_changed") != implementation_handoff.get("changed")
                    or implementation_handoff.get("changed") != []
                    or metadata.get("implementation_review_content_in_handoff") != implementation_handoff.get("content_in_handoff")
                    or implementation_handoff.get("content_in_handoff") is not True
                    or metadata.get("implementation_review_authorizes_execution") != implementation_handoff.get("authorizes_execution")
                    or implementation_handoff.get("authorizes_execution") is not False
                    or metadata.get("implementation_review_authorizes_completion_claim") != implementation_handoff.get("authorizes_completion_claim")
                    or implementation_handoff.get("authorizes_completion_claim") is not False
                    or metadata.get("implementation_review_approval_granted") != implementation_handoff.get("approval_granted")
                    or implementation_handoff.get("approval_granted") is not False
                    or metadata.get("implementation_review_boundaries") != handoff_boundaries
                    or metadata.get("implementation_review_next_safe_commands") != implementation_handoff.get("next_safe_commands")
                    or list(handoff_next_commands.values()) != implementation_handoff.get("next_safe_commands")
                    or len(handoff_next_commands) != 3
                    or implementation_handoff.get("connector") != metadata.get("connector")
                    or implementation_handoff.get("action") != metadata.get("action")
                    or implementation_handoff.get("tool_name") != metadata.get("tool_name")
                    or implementation_handoff.get("review_state") != metadata.get("review_state")
                    or implementation_handoff.get("review_ready") is not expected_review_ready
                    or implementation_handoff.get("implementation_review_allowed") is not expected_review_ready
                    or implementation_handoff.get("can_open_code_review_after_receipt") is not expected_review_ready
                    or implementation_handoff.get("route_blocked_until_implementation_review") is not (not expected_review_ready)
                    or implementation_handoff.get("review_checks") != metadata.get("review_checks")
                    or implementation_handoff.get("review_check_count") != len(metadata.get("review_checks") or [])
                    or implementation_handoff.get("passed_review_checks") != metadata.get("passed_review_checks")
                    or implementation_handoff.get("missing_fields") != metadata.get("missing_fields")
                    or handoff_receipt.get("id") != metadata.get("review_receipt_id")
                    or handoff_receipt.get("integration_scope_hash") != metadata.get("integration_scope_hash")
                    or handoff_receipt.get("contract_ready") is not expected_review_ready
                    or handoff_receipt.get("contract_row_count") != len(receipt_rows)
                    or handoff_receipt.get("contract_rows") != receipt_rows
                    or handoff_receipt.get("contract_summary") != metadata.get("implementation_review_receipt_contract_summary")
                    or handoff_receipt.get("authorizes_account_access")
                    or handoff_receipt.get("authorizes_route_unlock")
                    or handoff_receipt.get("reusable_for_other_scope")
                    or handoff_commands.get("proof_bundle") != metadata.get("proof_bundle_command")
                    or handoff_commands.get("implementation_review") != metadata.get("implementation_review_command")
                    or handoff_commands.get("next_safe") != metadata.get("next_safe_command")
                    or handoff_next_commands != handoff_commands
                    or handoff_queue.get("items") != review_proof_queue
                    or handoff_queue.get("count") != len(review_proof_queue)
                    or handoff_queue.get("next") != metadata.get("next_proof_command")
                    or handoff_queue.get("next_required") != metadata.get("next_required_command")
                    or handoff_queue.get("next_proof") != metadata.get("next_proof_command")
                    or handoff_states.get("proof_bundle_state") != metadata.get("proof_bundle_state")
                    or handoff_states.get("preflight_state") != metadata.get("preflight_state")
                    or handoff_states.get("spec_state") != metadata.get("spec_state")
                    or handoff_states.get("status_evidence_present") != metadata.get("status_evidence_present")
                    or handoff_states.get("metadata_row_contract_ready") != metadata.get("metadata_row_contract_ready")
                    or handoff_risk.get("suggested_risk") != metadata.get("suggested_risk")
                    or handoff_risk.get("approval_required") != metadata.get("approval_required")
                    or implementation_handoff.get("next_route_lock_requires_fresh_review") is not True
                    or handoff_boundaries.get("natural_language_routing_enabled")
                    or handoff_boundaries.get("adapter_default_state") != "disabled"
                    or handoff_boundaries.get("calls_external_service")
                    or handoff_boundaries.get("reads_personal_data")
                    or handoff_boundaries.get("executes_side_effect")
                    or handoff_boundaries.get("writes_memory")
                    or handoff_boundaries.get("authorizes_account_access")
                    or handoff_boundaries.get("authorizes_route_unlock")
                    or handoff_boundaries.get("authorizes_natural_language_routing")
                    or handoff_boundaries.get("authorizes_personal_data_read")
                    or handoff_boundaries.get("authorizes_side_effect")
                    or handoff_boundaries.get("authorizes_approval")
                    or handoff_boundaries.get("authorizes_model_call")
                    or handoff_boundaries.get("authorizes_tool_execution")
                    or handoff_boundaries.get("authorizes_external_service")
                    or handoff_boundaries.get("authorizes_execution")
                    or handoff_boundaries.get("authorizes_completion_claim")
                    or handoff_boundaries.get("approval_granted")
                    or handoff_boundaries.get("reusable_for_other_scope")
                ):
                    raise SystemExit(f"Integration implementation review handoff diverged from safety contract: {metadata}")
                for command in [
                    "integration proof bundle:",
                    "integration preflight contract:",
                    "integration implementation spec:",
                    "execution health report",
                ]:
                    if not any(str(item).startswith(command) for item in review_proof_queue):
                        raise SystemExit(f"Integration implementation review proof queue missed {command}: {metadata}")
                if "search mailbox metadata" in case and "status status api smoke passed" in case:
                    assert_contains(
                        result.response,
                        [
                            "Review state: IMPLEMENTATION_REVIEW_READY",
                            "Suggested risk: PERSONAL_DATA",
                            "proof bundle: pass",
                            "metadata row contract: pass",
                            "preflight contract: pass",
                            "implementation spec: pass",
                            "status visibility: pass",
                            "missing blockers: none",
                            "metadata row schema: id, timestamp, label, source",
                            "metadata row limit: 2",
                        "review receipt id:",
                        "scope hash:",
                        "implementation review command:",
                        "receipt contract summary: exact-scope-only, code-review-only, account-access-not-authorized, route-unlock-not-authorized, fresh-route-lock-review-required",
                        "code-review unlock: ready for this exact scope",
                        "natural-language auto-routing remains disabled",
                        "adapter default state remains disabled",
                            "next required command: `execution health report`",
                        ],
                        case,
                    )
                    if "next proof command: `execution health report`" in result.response:
                        raise SystemExit("Integration implementation review should render next required command, not next proof command.")
                    if (
                        metadata.get("review_state") != "IMPLEMENTATION_REVIEW_READY"
                        or metadata.get("missing_fields")
                        or metadata.get("passed_review_checks") != 5
                        or metadata.get("review_check_count") != 5
                        or metadata.get("suggested_risk") != "PERSONAL_DATA"
                        or metadata.get("sample_row_schema") != ["id", "timestamp", "label", "source"]
                        or metadata.get("sample_row_limit") != 2
                        or metadata.get("metadata_row_contract_ready") is not True
                        or metadata.get("natural_language_routing_enabled")
                        or metadata.get("adapter_default_state") != "disabled"
                        or not metadata.get("status_evidence_present")
                        or not metadata.get("implementation_review_allowed")
                        or metadata.get("implementation_review_receipt_contract_ready") is not True
                        or "code-review-only" not in metadata.get("implementation_review_receipt_contract_summary", [])
                        or metadata.get("route_blocked_until_implementation_review")
                        or metadata.get("can_open_code_review_after_receipt") is not True
                        or not metadata.get("review_receipt_id")
                        or len(str(metadata.get("integration_scope_hash") or "")) != 16
                        or not str(metadata.get("proof_bundle_command") or "").startswith("integration proof bundle:")
                        or not str(metadata.get("implementation_review_command") or "").startswith("integration implementation review:")
                        or metadata.get("next_proof_command") != "execution health report"
                        or metadata.get("next_safe_command") != "execution health report"
                    ):
                        raise SystemExit(f"Expected ready implementation review metadata: {metadata}")
                if "search mailbox metadata" in case and "status status api smoke passed" not in case:
                    assert_contains(result.response, ["Review state: IMPLEMENTATION_REVIEW_BLOCKED", "status visibility: blocked", "status/API smoke evidence"], case)
                    if (
                        metadata.get("review_state") != "IMPLEMENTATION_REVIEW_BLOCKED"
                        or "status/API smoke evidence" not in metadata.get("missing_fields", [])
                        or metadata.get("status_evidence_present")
                        or metadata.get("implementation_review_allowed")
                        or metadata.get("implementation_review_handoff_ready") is not True
                        or metadata.get("implementation_review_ready_for_operator") is not True
                        or metadata.get("implementation_review_receipt_contract_ready")
                        or "code-review-blocked" not in metadata.get("implementation_review_receipt_contract_summary", [])
                        or metadata.get("next_proof_command") != review_proof_queue[0]
                    ):
                        raise SystemExit(f"Expected status-blocked implementation review metadata: {metadata}")
                if "logged-in page" in case:
                    assert_contains(result.response, ["Review state: IMPLEMENTATION_REVIEW_BLOCKED", "proof bundle: blocked", "preflight contract: blocked", "implementation spec: blocked", "missing blockers:"], case)
                    if (
                        metadata.get("review_state") != "IMPLEMENTATION_REVIEW_BLOCKED"
                        or not metadata.get("missing_fields")
                        or metadata.get("implementation_review_allowed")
                    ):
                        raise SystemExit(f"Expected blocked implementation review metadata: {metadata}")
            elif "proof bundle" in case:
                assert_contains(
                    result.response,
                    [
                        "Jarvis personal integration proof bundle",
                        "This is read-only",
                        "Bundle state",
                        "Exact scope",
                        "Proof chain",
                        "metadata preview",
                        "disabled adapter acceptance",
                        "metadata row contract",
                        "enablement gate",
                        "rehearsal receipt",
                        "Implementation review posture",
                        "receipt contract rows:",
                        "receipt contract summary:",
                        "proof queue count",
                        "proof queue:",
                        "Stop conditions",
                    ],
                    case,
                )
                metadata = result.tool_results[0].metadata
                assert_read_only_metadata(metadata, "Integration proof bundle")
                proof_queue = metadata.get("proof_queue") or []
                if not proof_queue or metadata.get("proof_queue_count") != len(proof_queue):
                    raise SystemExit(f"Integration proof bundle missed proof queue metadata: {metadata}")
                if metadata.get("next_proof_command") not in proof_queue and metadata.get("next_proof_command") != "execution health report":
                    raise SystemExit(f"Integration proof bundle missed next proof command: {metadata}")
                if metadata.get("next_required_command") != metadata.get("next_proof_command"):
                    raise SystemExit(f"Integration proof bundle next required alias diverged: {metadata}")
                if metadata.get("integration_proof_queue") != proof_queue:
                    raise SystemExit(f"Integration proof bundle proof queue alias diverged: {metadata}")
                if metadata.get("integration_proof_queue_count") != len(proof_queue):
                    raise SystemExit(f"Integration proof bundle proof queue count alias diverged: {metadata}")
                if metadata.get("integration_next_required_command") != metadata.get("next_required_command"):
                    raise SystemExit(f"Integration proof bundle next required integration alias diverged: {metadata}")
                if metadata.get("integration_next_proof_command") != metadata.get("next_proof_command"):
                    raise SystemExit(f"Integration proof bundle next proof alias diverged: {metadata}")
                receipt_rows = metadata.get("implementation_review_receipt_contract_rows") or []
                if (
                    metadata.get("implementation_review_receipt_contract_row_count") != len(receipt_rows)
                    or len(receipt_rows) != 5
                    or metadata.get("implementation_review_receipt_contract_ready")
                    or metadata.get("implementation_review_receipt_authorizes_account_access")
                    or metadata.get("implementation_review_receipt_authorizes_route_unlock")
                    or metadata.get("implementation_review_receipt_reusable_for_other_scope")
                    or any(row.get("authorizes_account_access") or row.get("authorizes_route_unlock") or row.get("reusable_for_other_scope") for row in receipt_rows)
                ):
                    raise SystemExit(f"Integration proof bundle receipt contract unsafe: {metadata}")
                proof_bundle_handoff = metadata.get("proof_bundle_handoff") or {}
                handoff_receipt = proof_bundle_handoff.get("review_receipt") or {}
                handoff_queue = proof_bundle_handoff.get("proof_queue") or {}
                handoff_commands = proof_bundle_handoff.get("commands") or {}
                handoff_states = proof_bundle_handoff.get("source_states") or {}
                handoff_risk = proof_bundle_handoff.get("risk") or {}
                handoff_boundaries = proof_bundle_handoff.get("boundaries") or {}
                handoff_next_commands = proof_bundle_handoff.get("next_commands") or {}
                expected_bundle_ready = metadata.get("bundle_state") == "PROOF_BUNDLE_READY_FOR_IMPLEMENTATION_REVIEW"
                if (
                    metadata.get("proof_bundle_handoff_ready") != proof_bundle_handoff.get("handoff_ready")
                    or proof_bundle_handoff.get("proof_bundle_handoff_ready") != proof_bundle_handoff.get("handoff_ready")
                    or proof_bundle_handoff.get("handoff_ready") is not True
                    or metadata.get("proof_bundle_ready_for_operator") != proof_bundle_handoff.get("ready_for_operator")
                    or proof_bundle_handoff.get("ready_for_operator") is not True
                    or metadata.get("proof_bundle_state_changed") != proof_bundle_handoff.get("state_changed")
                    or proof_bundle_handoff.get("state_changed") is not False
                    or metadata.get("proof_bundle_changed") != proof_bundle_handoff.get("changed")
                    or proof_bundle_handoff.get("changed") != []
                    or metadata.get("proof_bundle_content_in_handoff") != proof_bundle_handoff.get("content_in_handoff")
                    or proof_bundle_handoff.get("content_in_handoff") is not True
                    or metadata.get("proof_bundle_authorizes_execution") != proof_bundle_handoff.get("authorizes_execution")
                    or proof_bundle_handoff.get("authorizes_execution") is not False
                    or metadata.get("proof_bundle_authorizes_completion_claim") != proof_bundle_handoff.get("authorizes_completion_claim")
                    or proof_bundle_handoff.get("authorizes_completion_claim") is not False
                    or metadata.get("proof_bundle_approval_granted") != proof_bundle_handoff.get("approval_granted")
                    or proof_bundle_handoff.get("approval_granted") is not False
                    or metadata.get("proof_bundle_boundaries") != handoff_boundaries
                    or metadata.get("proof_bundle_next_safe_commands") != proof_bundle_handoff.get("next_safe_commands")
                    or list(handoff_next_commands.values()) != proof_bundle_handoff.get("next_safe_commands")
                    or len(handoff_next_commands) != 3
                    or proof_bundle_handoff.get("connector") != metadata.get("connector")
                    or proof_bundle_handoff.get("action") != metadata.get("action")
                    or proof_bundle_handoff.get("tool_name") != metadata.get("tool_name")
                    or proof_bundle_handoff.get("bundle_state") != metadata.get("bundle_state")
                    or proof_bundle_handoff.get("bundle_ready_for_implementation_review") is not expected_bundle_ready
                    or proof_bundle_handoff.get("implementation_review_allowed") is not expected_bundle_ready
                    or proof_bundle_handoff.get("can_open_code_review_after_receipt")
                    or proof_bundle_handoff.get("route_blocked_until_implementation_review") is not True
                    or proof_bundle_handoff.get("proof_checks") != metadata.get("proof_checks")
                    or proof_bundle_handoff.get("proof_check_count") != len(metadata.get("proof_checks") or [])
                    or proof_bundle_handoff.get("passed_proof_checks") != metadata.get("passed_proof_checks")
                    or proof_bundle_handoff.get("missing_fields") != metadata.get("missing_fields")
                    or handoff_receipt.get("id") != metadata.get("review_receipt_id")
                    or handoff_receipt.get("integration_scope_hash") != metadata.get("integration_scope_hash")
                    or handoff_receipt.get("contract_ready")
                    or handoff_receipt.get("contract_row_count") != len(receipt_rows)
                    or handoff_receipt.get("contract_rows") != receipt_rows
                    or handoff_receipt.get("contract_summary") != metadata.get("implementation_review_receipt_contract_summary")
                    or handoff_receipt.get("authorizes_account_access")
                    or handoff_receipt.get("authorizes_route_unlock")
                    or handoff_receipt.get("reusable_for_other_scope")
                    or handoff_commands.get("proof_bundle") != metadata.get("proof_bundle_command")
                    or handoff_commands.get("implementation_review") != metadata.get("implementation_review_command")
                    or handoff_commands.get("next_safe") != metadata.get("next_safe_command")
                    or handoff_next_commands != handoff_commands
                    or handoff_queue.get("items") != proof_queue
                    or handoff_queue.get("count") != len(proof_queue)
                    or handoff_queue.get("next") != metadata.get("next_proof_command")
                    or handoff_queue.get("next_required") != metadata.get("next_required_command")
                    or handoff_queue.get("next_proof") != metadata.get("next_proof_command")
                    or handoff_states.get("metadata_preview_state") != metadata.get("metadata_preview_state")
                    or handoff_states.get("adapter_acceptance_state") != metadata.get("adapter_acceptance_state")
                    or handoff_states.get("enablement_verdict") != metadata.get("enablement_verdict")
                    or handoff_states.get("rehearsal_state") != metadata.get("rehearsal_state")
                    or handoff_states.get("metadata_row_contract_ready") != metadata.get("metadata_row_contract_ready")
                    or handoff_risk.get("suggested_risk") != metadata.get("suggested_risk")
                    or handoff_risk.get("approval_required") != metadata.get("approval_required")
                    or proof_bundle_handoff.get("next_step_requires_implementation_review") is not True
                    or proof_bundle_handoff.get("next_route_lock_requires_fresh_review") is not True
                    or handoff_boundaries.get("natural_language_routing_enabled")
                    or handoff_boundaries.get("calls_external_service")
                    or handoff_boundaries.get("reads_personal_data")
                    or handoff_boundaries.get("executes_side_effect")
                    or handoff_boundaries.get("writes_memory")
                    or handoff_boundaries.get("authorizes_account_access")
                    or handoff_boundaries.get("authorizes_route_unlock")
                    or handoff_boundaries.get("authorizes_natural_language_routing")
                    or handoff_boundaries.get("authorizes_personal_data_read")
                    or handoff_boundaries.get("authorizes_side_effect")
                    or handoff_boundaries.get("authorizes_approval")
                    or handoff_boundaries.get("authorizes_model_call")
                    or handoff_boundaries.get("authorizes_tool_execution")
                    or handoff_boundaries.get("authorizes_external_service")
                    or handoff_boundaries.get("authorizes_execution")
                    or handoff_boundaries.get("authorizes_completion_claim")
                    or handoff_boundaries.get("approval_granted")
                    or handoff_boundaries.get("reusable_for_other_scope")
                ):
                    raise SystemExit(f"Integration proof bundle handoff diverged from safety contract: {metadata}")
                for command in [
                    "integration metadata preview:",
                    "integration adapter acceptance:",
                    "integration adapter manifest:",
                    "integration enablement gate:",
                    "integration rehearsal receipt:",
                    "execution health report",
                    "integration implementation review:",
                ]:
                    if not any(str(item).startswith(command) for item in proof_queue):
                        raise SystemExit(f"Integration proof bundle proof queue missed {command}: {metadata}")
                if "search mailbox metadata" in case:
                    assert_contains(
                        result.response,
                        [
                            "Bundle state: PROOF_BUNDLE_READY_FOR_IMPLEMENTATION_REVIEW",
                            "Suggested risk: PERSONAL_DATA",
                            "metadata preview: pass",
                            "disabled adapter acceptance: pass",
                            "metadata row contract: pass",
                            "enablement gate: pass",
                            "rehearsal receipt: pass",
                            "missing blockers: none",
                            "metadata row schema: id, timestamp, label, source",
                            "metadata row limit: 2",
                            "review receipt id:",
                            "scope hash:",
                            "implementation review command:",
                            "receipt contract summary: proof-bundle-only, implementation-review-required, account-access-not-authorized, route-unlock-not-authorized, fresh-scope-review-required",
                            "route blocker:",
                            "next required command: `execution health report`",
                            "natural-language auto-routing: disabled",
                        ],
                        case,
                    )
                    if "next proof command: `execution health report`" in result.response:
                        raise SystemExit("Integration proof bundle should render next required command, not next proof command.")
                    if (
                        metadata.get("bundle_state") != "PROOF_BUNDLE_READY_FOR_IMPLEMENTATION_REVIEW"
                        or metadata.get("missing_fields")
                        or metadata.get("passed_proof_checks") != 5
                        or metadata.get("proof_check_count") != 5
                        or metadata.get("suggested_risk") != "PERSONAL_DATA"
                        or metadata.get("sample_row_schema") != ["id", "timestamp", "label", "source"]
                        or metadata.get("sample_row_limit") != 2
                        or metadata.get("metadata_row_contract_ready") is not True
                        or metadata.get("natural_language_routing_enabled")
                        or not metadata.get("implementation_review_allowed")
                        or metadata.get("route_blocked_until_implementation_review") is not True
                        or metadata.get("can_open_code_review_after_receipt")
                        or "proof-bundle-only" not in metadata.get("implementation_review_receipt_contract_summary", [])
                        or not metadata.get("review_receipt_id")
                        or len(str(metadata.get("integration_scope_hash") or "")) != 16
                        or not str(metadata.get("proof_bundle_command") or "").startswith("integration proof bundle:")
                        or not str(metadata.get("implementation_review_command") or "").startswith("integration implementation review:")
                        or metadata.get("next_proof_command") != "execution health report"
                        or metadata.get("next_safe_command") != "execution health report"
                    ):
                        raise SystemExit(f"Expected ready proof bundle metadata: {metadata}")
                if "logged-in page" in case:
                    assert_contains(result.response, ["Bundle state: PROOF_BUNDLE_BLOCKED", "metadata preview: blocked", "enablement gate: blocked", "missing blockers:"], case)
                    if (
                        metadata.get("bundle_state") != "PROOF_BUNDLE_BLOCKED"
                        or not metadata.get("missing_fields")
                        or metadata.get("implementation_review_allowed")
                        or metadata.get("proof_bundle_handoff_ready") is not True
                        or metadata.get("proof_bundle_ready_for_operator") is not True
                        or metadata.get("next_proof_command") != proof_queue[0]
                    ):
                        raise SystemExit(f"Expected blocked proof bundle metadata: {metadata}")
            elif "metadata preview" in case:
                assert_contains(
                    result.response,
                    [
                        "Jarvis personal integration metadata preview",
                        "This is read-only",
                        "Preview state",
                        "Exact metadata scope",
                        "Fake preview rows",
                        "Row schema contract",
                        "allowed fields: id, timestamp, label, source",
                        "row limit: 2",
                        "Adapter contract",
                        "Proof carried into implementation",
                        "Known connector boundaries",
                        "Stop conditions",
                    ],
                    case,
                )
                metadata = result.tool_results[0].metadata
                assert_read_only_metadata(metadata, "Integration metadata preview")
                handoff = metadata.get("metadata_preview_handoff")
                if not isinstance(handoff, dict):
                    raise SystemExit(f"Expected structured metadata preview handoff: {metadata}")
                handoff_scope = handoff.get("scope") or {}
                handoff_proof = handoff.get("proof") or {}
                handoff_rows = handoff.get("sample_rows") or []
                handoff_contract = handoff.get("row_contract") or {}
                handoff_risk = handoff.get("risk") or {}
                handoff_boundaries = handoff.get("boundaries") or {}
                if (
                    metadata.get("metadata_preview_handoff_ready") != metadata.get("metadata_preview_ready")
                    or handoff.get("metadata_preview_handoff_ready") != metadata.get("metadata_preview_ready")
                    or metadata.get("metadata_preview_handoff_ready") != handoff.get("handoff_ready")
                    or metadata.get("metadata_preview_ready_for_operator") != handoff.get("ready_for_operator")
                    or metadata.get("metadata_preview_ready_for_operator") != metadata.get("metadata_preview_ready")
                    or metadata.get("metadata_preview_state_changed") is not False
                    or metadata.get("metadata_preview_changed") != []
                    or metadata.get("metadata_preview_content_in_handoff") is not True
                    or metadata.get("metadata_preview_authorizes_execution")
                    or metadata.get("metadata_preview_authorizes_completion_claim")
                    or metadata.get("metadata_preview_approval_granted")
                    or metadata.get("metadata_preview_boundaries") != handoff_boundaries
                    or metadata.get("metadata_preview_next_safe_commands") != handoff.get("next_safe_commands")
                    or not metadata.get("metadata_preview_next_safe_commands")
                    or handoff.get("state_changed") is not False
                    or handoff.get("changed") != []
                    or handoff.get("content_in_handoff") is not True
                    or handoff.get("authorizes_execution")
                    or handoff.get("authorizes_completion_claim")
                    or handoff.get("approval_granted")
                    or handoff.get("connector") != metadata.get("connector")
                    or handoff.get("requested_connector") != metadata.get("requested_connector")
                    or handoff.get("action") != metadata.get("action")
                    or handoff.get("tool_name") != metadata.get("tool_name")
                    or handoff.get("preview_state") != metadata.get("preview_state")
                    or handoff.get("metadata_preview_ready") != metadata.get("metadata_preview_ready")
                    or handoff.get("metadata_only") != metadata.get("metadata_only")
                    or handoff.get("adapter_state") != metadata.get("adapter_state")
                    or handoff.get("missing_fields") != metadata.get("missing_fields")
                    or handoff_scope.get("target") != metadata.get("target")
                    or handoff_scope.get("time_range") != metadata.get("time_range")
                    or handoff_scope.get("data_level") != metadata.get("data_level")
                    or handoff_scope.get("verification") != metadata.get("verification")
                    or handoff_proof.get("tests") != metadata.get("tests")
                    or handoff_proof.get("audit") != metadata.get("audit")
                    or handoff_proof.get("acceptance") != metadata.get("acceptance")
                    or handoff_rows != metadata.get("sample_rows")
                    or handoff.get("sample_row_count") != metadata.get("sample_row_count")
                    or handoff_contract.get("sample_row_schema") != metadata.get("sample_row_schema")
                    or handoff_contract.get("sample_row_schema_fields") != metadata.get("sample_row_schema_fields")
                    or handoff_contract.get("sample_row_limit") != metadata.get("sample_row_limit")
                    or handoff_contract.get("sample_rows_bounded") != metadata.get("sample_rows_bounded")
                    or handoff_contract.get("blocked_payload_fields") != metadata.get("blocked_payload_fields")
                    or handoff_risk.get("suggested_risk") != metadata.get("suggested_risk")
                    or handoff_risk.get("approval_required") != metadata.get("approval_required")
                    or handoff.get("next_step_requires_proof_bundle") is not True
                    or handoff.get("next_route_lock_requires_fresh_review") is not True
                    or handoff_boundaries.get("natural_language_routing_enabled")
                    or not handoff_boundaries.get("adapter_call_skipped")
                    or handoff_boundaries.get("calls_external_service")
                    or handoff_boundaries.get("reads_personal_data")
                    or handoff_boundaries.get("executes_side_effect")
                    or handoff_boundaries.get("authorizes_account_access")
                    or handoff_boundaries.get("authorizes_route_unlock")
                    or handoff_boundaries.get("authorizes_natural_language_routing")
                    or handoff_boundaries.get("authorizes_personal_data_read")
                    or handoff_boundaries.get("authorizes_side_effect")
                    or handoff_boundaries.get("authorizes_approval")
                    or handoff_boundaries.get("authorizes_execution")
                    or handoff_boundaries.get("authorizes_completion_claim")
                    or handoff_boundaries.get("approval_granted")
                    or handoff_boundaries.get("authorizes_model_call")
                    or handoff_boundaries.get("authorizes_tool_execution")
                    or handoff_boundaries.get("authorizes_external_service")
                    or handoff_boundaries.get("reusable_for_other_scope")
                ):
                    raise SystemExit(f"Metadata preview handoff diverged from flat metadata or safety contract: {metadata}")
                if "data metadata-only" in case and "send draft reply" not in case:
                    assert_contains(
                        result.response,
                        [
                            "Preview state: METADATA_PREVIEW_READY",
                            "Metadata clamp: active",
                            "Adapter state: metadata adapter disabled",
                            "missing blockers: none",
                            "sample-1: id, timestamp, label, source only",
                            "no email bodies, message text, calendar notes",
                        ],
                        case,
                    )
                    if (
                        metadata.get("preview_state") != "METADATA_PREVIEW_READY"
                        or metadata.get("missing_fields")
                        or not metadata.get("metadata_only")
                        or metadata.get("adapter_state") != "metadata adapter disabled"
                        or metadata.get("sample_row_count") != 2
                        or metadata.get("sample_row_schema") != ["id", "timestamp", "label", "source"]
                        or metadata.get("sample_row_limit") != 2
                        or metadata.get("sample_rows_bounded") is not True
                        or metadata.get("natural_language_routing_enabled")
                        or not metadata.get("metadata_preview_ready")
                    ):
                        raise SystemExit(f"Expected ready metadata preview metadata: {metadata}")
                if "data full body" in case:
                    assert_contains(result.response, ["Preview state: METADATA_PREVIEW_BLOCKED", "metadata-only data level"], case)
                    if metadata.get("preview_state") != "METADATA_PREVIEW_BLOCKED" or metadata.get("metadata_preview_ready"):
                        raise SystemExit(f"Expected blocked full-content metadata preview: {metadata}")
                if "send draft reply" in case:
                    assert_contains(result.response, ["Preview state: METADATA_PREVIEW_BLOCKED", "non-side-effect action"], case)
                    if metadata.get("preview_state") != "METADATA_PREVIEW_BLOCKED" or metadata.get("metadata_preview_ready"):
                        raise SystemExit(f"Expected blocked side-effect metadata preview: {metadata}")
            elif "contract" in case:
                assert_contains(
                    result.response,
                    [
                        "Jarvis personal integration boundary contract",
                        "Default state",
                        "Allowed without approval",
                        "Requires approval before reading personal data",
                        "Requires approval before side effects",
                        "Approval prompt template",
                        "Caching rules",
                        "Audit and rollback",
                        "Hard stops",
                    ],
                    case,
                )
                metadata = result.tool_results[0].metadata
                assert_read_only_metadata(metadata, "Integration contract")
                handoff = metadata.get("boundary_contract_handoff")
                if not isinstance(handoff, dict):
                    raise SystemExit(f"Expected structured boundary contract handoff: {metadata}")
                handoff_default = handoff.get("default_state") or {}
                handoff_approval = handoff.get("approval_required") or {}
                handoff_prompt = handoff.get("approval_prompt_template") or {}
                handoff_cache = handoff.get("caching_rules") or {}
                handoff_audit = handoff.get("audit_and_rollback") or {}
                handoff_commands = handoff.get("next_commands") or {}
                handoff_boundaries = handoff.get("boundaries") or {}
                if (
                    metadata.get("boundary_contract_handoff_ready") is not True
                    or handoff.get("boundary_contract_handoff_ready") is not True
                    or metadata.get("boundary_contract_handoff_ready") != handoff.get("handoff_ready")
                    or metadata.get("boundary_contract_ready_for_operator") is not True
                    or handoff.get("ready_for_operator") is not True
                    or metadata.get("boundary_contract_state_changed") is not False
                    or handoff.get("state_changed") is not False
                    or metadata.get("boundary_contract_changed") != []
                    or handoff.get("changed") != []
                    or metadata.get("boundary_contract_content_in_handoff") is not True
                    or handoff.get("content_in_handoff") is not True
                    or metadata.get("boundary_contract_authorizes_execution") is not False
                    or metadata.get("boundary_contract_authorizes_completion_claim") is not False
                    or metadata.get("boundary_contract_approval_granted") is not False
                    or handoff.get("authorizes_execution") is not False
                    or handoff.get("authorizes_completion_claim") is not False
                    or handoff.get("approval_granted") is not False
                    or metadata.get("boundary_contract_boundaries") != handoff_boundaries
                    or metadata.get("boundary_contract_next_safe_commands") != handoff.get("next_safe_commands")
                    or set((handoff.get("next_commands") or {}).values()) != set(metadata.get("boundary_contract_next_safe_commands") or [])
                    or handoff.get("connector") != metadata.get("connector")
                    or handoff.get("requested_connector") != metadata.get("requested_connector")
                    or handoff_default.get("connected_by_default")
                    or handoff_default.get("account_tokens_available")
                    or handoff_default.get("cookies_available")
                    or handoff_default.get("mailbox_calendar_message_or_contact_data_available")
                    or handoff_default.get("first_implementation_should_be_read_only_and_narrow") is not True
                    or not handoff.get("allowed_without_approval")
                    or handoff_approval.get("personal_data") != metadata.get("personal_data_surface_items")
                    or handoff_approval.get("side_effects") != metadata.get("side_effect_surface_items")
                    or handoff_approval.get("personal_data_count") != metadata.get("personal_data_surfaces")
                    or handoff_approval.get("side_effect_count") != metadata.get("side_effect_surfaces")
                    or handoff.get("read_only_candidates") != metadata.get("read_only_candidate_items")
                    or handoff.get("read_only_candidate_count") != metadata.get("read_only_candidates")
                    or handoff_prompt.get("future_tool") != f"future_{metadata.get('connector')}_tool"
                    or handoff_prompt.get("one_shot") is not True
                    or handoff_prompt.get("cannot_transfer_scope") is not True
                    or "full email bodies" not in (handoff_cache.get("blocked_by_default") or [])
                    or handoff_cache.get("source_and_timestamp_required_for_saved_summary") is not True
                    or handoff_audit.get("tool_run_audit_required") is not True
                    or handoff_audit.get("side_effect_pending_approval_receipt_required") is not True
                    or handoff_audit.get("read_only_preview_required_before_mutation") is not True
                    or handoff_audit.get("cancel_or_dismiss_path_required_before_mutation") is not True
                    or handoff_commands.get("migration_plan") != f"integration migration plan: {metadata.get('connector')}"
                    or handoff_commands.get("scope_packet") != f"integration scope packet: {metadata.get('connector')} -> {(metadata.get('read_only_candidate_items') or [''])[0]}"
                    or handoff_boundaries.get("natural_language_routing_enabled")
                    or handoff_boundaries.get("calls_external_service")
                    or handoff_boundaries.get("reads_personal_data")
                    or handoff_boundaries.get("executes_side_effect")
                    or handoff_boundaries.get("writes_memory")
                    or handoff_boundaries.get("authorizes_account_access")
                    or handoff_boundaries.get("authorizes_route_unlock")
                    or handoff_boundaries.get("authorizes_natural_language_routing")
                    or handoff_boundaries.get("authorizes_personal_data_read")
                    or handoff_boundaries.get("authorizes_side_effect")
                    or handoff_boundaries.get("authorizes_approval")
                    or handoff_boundaries.get("authorizes_model_call")
                    or handoff_boundaries.get("authorizes_tool_execution")
                    or handoff_boundaries.get("authorizes_external_service")
                    or handoff_boundaries.get("authorizes_execution")
                    or handoff_boundaries.get("authorizes_completion_claim")
                    or handoff_boundaries.get("approval_granted")
                    or handoff_boundaries.get("reusable_for_other_scope")
                    or handoff.get("next_step_requires_action_preview") is not True
                    or handoff.get("next_step_requires_scope_packet") is not True
                    or handoff.get("next_route_lock_requires_fresh_review") is not True
                ):
                    raise SystemExit(f"Boundary contract handoff diverged from flat metadata or safety contract: {metadata}")
            elif "action preview" in case:
                assert_contains(
                    result.response,
                    [
                        "Jarvis integration action preview",
                        "This is read-only",
                        "Connector:",
                        "Suggested risk:",
                        "Approval required:",
                        "Known connector surfaces",
                        "Safe preview packet",
                        "Hard stops",
                    ],
                    case,
                )
                metadata = result.tool_results[0].metadata
                assert_read_only_metadata(metadata, "Integration action preview")
                handoff = metadata.get("action_preview_handoff")
                if not isinstance(handoff, dict):
                    raise SystemExit(f"Expected structured action preview handoff: {metadata}")
                handoff_classification = handoff.get("classification") or {}
                handoff_preview = handoff.get("safe_preview_packet") or {}
                handoff_commands = handoff.get("next_commands") or {}
                handoff_stops = handoff.get("stop_conditions") or {}
                handoff_boundaries = handoff.get("boundaries") or {}
                if (
                    metadata.get("action_preview_handoff_ready") is not True
                    or handoff.get("action_preview_handoff_ready") is not True
                    or metadata.get("action_preview_handoff_ready") != handoff.get("handoff_ready")
                    or metadata.get("action_preview_ready_for_operator") is not True
                    or handoff.get("ready_for_operator") is not True
                    or metadata.get("action_preview_state_changed") is not False
                    or handoff.get("state_changed") is not False
                    or metadata.get("action_preview_changed") != []
                    or handoff.get("changed") != []
                    or metadata.get("action_preview_content_in_handoff") is not True
                    or handoff.get("content_in_handoff") is not True
                    or metadata.get("action_preview_authorizes_execution") is not False
                    or metadata.get("action_preview_authorizes_completion_claim") is not False
                    or metadata.get("action_preview_approval_granted") is not False
                    or handoff.get("authorizes_execution") is not False
                    or handoff.get("authorizes_completion_claim") is not False
                    or handoff.get("approval_granted") is not False
                    or metadata.get("action_preview_boundaries") != handoff_boundaries
                    or metadata.get("action_preview_next_safe_commands") != handoff.get("next_safe_commands")
                    or set((handoff.get("next_commands") or {}).values()) != set(metadata.get("action_preview_next_safe_commands") or [])
                    or handoff.get("connector") != metadata.get("connector")
                    or handoff.get("requested_connector") != metadata.get("requested_connector")
                    or handoff.get("action") != metadata.get("action")
                    or handoff_classification.get("suggested_risk") != metadata.get("suggested_risk")
                    or handoff_classification.get("approval_required") != metadata.get("approval_required")
                    or handoff_classification.get("requires_approval") != metadata.get("requires_approval")
                    or not handoff_classification.get("reason")
                    or handoff_preview.get("future_tool") != f"future_{metadata.get('connector')}_tool"
                    or handoff_preview.get("requested_action") != metadata.get("action")
                    or "metadata-only" not in handoff_preview.get("safer_alternative", "")
                    or handoff_commands.get("scope_packet") != f"integration scope packet: {metadata.get('connector')} -> {metadata.get('action')}"
                    or handoff_commands.get("dry_run_contract") != f"integration dry run contract: {metadata.get('connector')} -> {metadata.get('action')}"
                    or handoff_commands.get("boundary_contract") != f"integration boundary contract: {metadata.get('connector')}"
                    or handoff_stops.get("requires_explicit_per_action_approval_for_side_effect") is not True
                    or handoff_stops.get("requires_approval_chain_proof") is not True
                    or handoff_stops.get("requires_exact_source_scope_for_private_reads") is not True
                    or handoff_stops.get("stop_if_account_recipient_thread_event_or_date_range_ambiguous") is not True
                    or handoff_boundaries.get("natural_language_routing_enabled")
                    or handoff_boundaries.get("calls_external_service")
                    or handoff_boundaries.get("reads_personal_data")
                    or handoff_boundaries.get("executes_side_effect")
                    or handoff_boundaries.get("writes_memory")
                    or handoff_boundaries.get("authorizes_account_access")
                    or handoff_boundaries.get("authorizes_route_unlock")
                    or handoff_boundaries.get("authorizes_natural_language_routing")
                    or handoff_boundaries.get("authorizes_personal_data_read")
                    or handoff_boundaries.get("authorizes_side_effect")
                    or handoff_boundaries.get("authorizes_approval")
                    or handoff_boundaries.get("authorizes_model_call")
                    or handoff_boundaries.get("authorizes_tool_execution")
                    or handoff_boundaries.get("authorizes_external_service")
                    or handoff_boundaries.get("authorizes_execution")
                    or handoff_boundaries.get("authorizes_completion_claim")
                    or handoff_boundaries.get("approval_granted")
                    or handoff_boundaries.get("reusable_for_other_scope")
                    or handoff.get("next_step_requires_scope_packet") is not True
                    or handoff.get("next_step_requires_dry_run_contract") is not True
                    or handoff.get("next_route_lock_requires_fresh_review") is not True
                ):
                    raise SystemExit(f"Action preview handoff diverged from flat metadata or safety contract: {metadata}")
                if "search mailbox metadata" in case:
                    assert_contains(result.response, ["Suggested risk: PERSONAL_DATA", "Approval required: yes"], case)
                if "send draft reply" in case:
                    assert_contains(result.response, ["Suggested risk: EXTERNAL_SIDE_EFFECT", "Approval required: yes"], case)
                if "logged-in page" in case:
                    assert_contains(result.response, ["Suggested risk: PERSONAL_DATA", "Approval required: yes"], case)
            elif "scope packet" in case:
                assert_contains(
                    result.response,
                    [
                        "Jarvis personal integration scope packet",
                        "This is read-only",
                        "Scope fields",
                        "Connector boundary",
                        "Approval packet seed",
                        "Stop conditions",
                    ],
                    case,
                )
                metadata = result.tool_results[0].metadata
                assert_read_only_metadata(metadata, "Integration scope packet")
                handoff = metadata.get("scope_packet_handoff")
                if not isinstance(handoff, dict):
                    raise SystemExit(f"Expected structured scope packet handoff: {metadata}")
                handoff_scope = handoff.get("scope") or {}
                handoff_classification = handoff.get("classification") or {}
                handoff_approval = handoff.get("approval_packet_seed") or {}
                handoff_commands = handoff.get("next_commands") or {}
                handoff_stops = handoff.get("stop_conditions") or {}
                handoff_boundaries = handoff.get("boundaries") or {}
                if (
                    metadata.get("scope_packet_handoff_ready") is not True
                    or handoff.get("scope_packet_handoff_ready") is not True
                    or metadata.get("scope_packet_handoff_ready") != handoff.get("handoff_ready")
                    or metadata.get("scope_packet_ready_for_operator") is not True
                    or handoff.get("ready_for_operator") is not True
                    or metadata.get("scope_packet_state_changed") is not False
                    or handoff.get("state_changed") is not False
                    or metadata.get("scope_packet_changed") != []
                    or handoff.get("changed") != []
                    or metadata.get("scope_packet_content_in_handoff") is not True
                    or handoff.get("content_in_handoff") is not True
                    or metadata.get("scope_packet_authorizes_execution") is not False
                    or metadata.get("scope_packet_authorizes_completion_claim") is not False
                    or metadata.get("scope_packet_approval_granted") is not False
                    or handoff.get("authorizes_execution") is not False
                    or handoff.get("authorizes_completion_claim") is not False
                    or handoff.get("approval_granted") is not False
                    or metadata.get("scope_packet_boundaries") != handoff_boundaries
                    or metadata.get("scope_packet_next_safe_commands") != handoff.get("next_safe_commands")
                    or set((handoff.get("next_commands") or {}).values()) != set(metadata.get("scope_packet_next_safe_commands") or [])
                    or handoff.get("connector") != metadata.get("connector")
                    or handoff.get("requested_connector") != metadata.get("requested_connector")
                    or handoff.get("action") != metadata.get("action")
                    or handoff.get("scope_ready") != metadata.get("scope_ready")
                    or handoff.get("missing_fields") != metadata.get("missing_fields")
                    or handoff_scope.get("target") != metadata.get("target")
                    or handoff_scope.get("time_range") != metadata.get("time_range")
                    or handoff_scope.get("data_level") != metadata.get("data_level")
                    or handoff_classification.get("suggested_risk") != metadata.get("suggested_risk")
                    or handoff_classification.get("approval_required") != metadata.get("approval_required")
                    or handoff_classification.get("requires_approval") != metadata.get("requires_approval")
                    or not handoff_classification.get("reason")
                    or handoff_approval.get("future_tool") != f"future_{metadata.get('connector')}_tool"
                    or handoff_approval.get("action") != metadata.get("action")
                    or handoff_approval.get("connector") != metadata.get("connector")
                    or handoff_approval.get("target") != (metadata.get("target") or "<required before use>")
                    or handoff_approval.get("time_range") != (metadata.get("time_range") or "<required before use>")
                    or handoff_approval.get("one_shot") is not True
                    or handoff_approval.get("cannot_transfer_scope") is not True
                    or handoff_commands.get("action_preview") != f"integration action preview: {metadata.get('connector')} -> {metadata.get('action')}"
                    or handoff_stops.get("stop_if_account_recipient_thread_event_page_list_or_date_range_ambiguous") is not True
                    or handoff_stops.get("stop_if_mutating_action_without_fresh_approval_receipt") is not True
                    or handoff_stops.get("stop_if_private_source_scope_not_explicitly_approved") is not True
                    or handoff_stops.get("requires_approval_chain_proof_for_real_use") != metadata.get("requires_approval")
                    or handoff_boundaries.get("natural_language_routing_enabled")
                    or handoff_boundaries.get("calls_external_service")
                    or handoff_boundaries.get("reads_personal_data")
                    or handoff_boundaries.get("executes_side_effect")
                    or handoff_boundaries.get("writes_memory")
                    or handoff_boundaries.get("authorizes_account_access")
                    or handoff_boundaries.get("authorizes_route_unlock")
                    or handoff_boundaries.get("authorizes_natural_language_routing")
                    or handoff_boundaries.get("authorizes_personal_data_read")
                    or handoff_boundaries.get("authorizes_side_effect")
                    or handoff_boundaries.get("authorizes_approval")
                    or handoff_boundaries.get("authorizes_model_call")
                    or handoff_boundaries.get("authorizes_tool_execution")
                    or handoff_boundaries.get("authorizes_external_service")
                    or handoff_boundaries.get("authorizes_execution")
                    or handoff_boundaries.get("authorizes_completion_claim")
                    or handoff_boundaries.get("approval_granted")
                    or handoff_boundaries.get("reusable_for_other_scope")
                    or handoff.get("next_step_requires_dry_run_contract") != metadata.get("scope_ready")
                    or handoff.get("next_step_requires_runbook") != metadata.get("scope_ready")
                    or handoff.get("next_route_lock_requires_fresh_review") is not True
                ):
                    raise SystemExit(f"Scope packet handoff diverged from flat metadata or safety contract: {metadata}")
                if "metadata" in case:
                    assert_contains(result.response, ["target/source: inbox", "time range or selected item: last 7 days", "data level: metadata-only", "missing fields: none"], case)
                    if metadata.get("suggested_risk") != "PERSONAL_DATA":
                        raise SystemExit(f"Expected personal-data risk for scoped email search: {metadata}")
                if "create event" in case:
                    assert_contains(result.response, ["Suggested risk: EXTERNAL_SIDE_EFFECT", "Approval required: yes"], case)
                    if metadata.get("suggested_risk") != "EXTERNAL_SIDE_EFFECT":
                        raise SystemExit(f"Expected side-effect risk for scoped calendar creation: {metadata}")
                if "logged-in page" in case:
                    assert_contains(result.response, ["missing fields:", "target/source", "time range or selected item"], case)
            elif "plan" in case:
                assert_contains(
                    result.response,
                    [
                        "Jarvis personal integration migration plan",
                        "Phase 1 - boundary contract",
                        "Phase 2 - risk mapping",
                        "Phase 3 - implementation checklist",
                        "Phase 4 - verification",
                        "PERSONAL_DATA",
                        "EXTERNAL_SIDE_EFFECT",
                        "pending approval queue",
                        "Hard safety rule",
                    ],
                    case,
                )
                metadata = result.tool_results[0].metadata
                assert_read_only_metadata(metadata, "Integration migration plan")
                handoff = metadata.get("migration_plan_handoff")
                if not isinstance(handoff, dict):
                    raise SystemExit(f"Expected structured migration plan handoff: {metadata}")
                handoff_risk = handoff.get("risk_map") or {}
                handoff_commands = handoff.get("next_commands") or {}
                handoff_requirements = handoff.get("proof_requirements") or {}
                handoff_boundaries = handoff.get("boundaries") or {}
                if (
                    metadata.get("migration_plan_handoff_ready") is not True
                    or handoff.get("migration_plan_handoff_ready") is not True
                    or metadata.get("migration_plan_handoff_ready") != handoff.get("handoff_ready")
                    or metadata.get("migration_plan_ready_for_operator") is not True
                    or handoff.get("ready_for_operator") is not True
                    or metadata.get("migration_plan_state_changed") is not False
                    or handoff.get("state_changed") is not False
                    or metadata.get("migration_plan_changed") != []
                    or handoff.get("changed") != []
                    or metadata.get("migration_plan_content_in_handoff") is not True
                    or handoff.get("content_in_handoff") is not True
                    or metadata.get("migration_plan_authorizes_execution") is not False
                    or metadata.get("migration_plan_authorizes_completion_claim") is not False
                    or metadata.get("migration_plan_approval_granted") is not False
                    or handoff.get("authorizes_execution") is not False
                    or handoff.get("authorizes_completion_claim") is not False
                    or handoff.get("approval_granted") is not False
                    or metadata.get("migration_plan_boundaries") != handoff_boundaries
                    or metadata.get("migration_plan_next_safe_commands") != handoff.get("next_safe_commands")
                    or set((handoff.get("next_commands") or {}).values()) != set(metadata.get("migration_plan_next_safe_commands") or [])
                    or handoff.get("connector") != metadata.get("connector")
                    or handoff.get("requested_connector") != metadata.get("requested_connector")
                    or handoff.get("phase_count") != len(handoff.get("phases") or [])
                    or handoff.get("phase_count") != 4
                    or handoff_risk.get("read_only_candidates") != metadata.get("read_only_candidates")
                    or handoff_risk.get("personal_data_candidates") != metadata.get("personal_data_candidates")
                    or handoff_risk.get("side_effect_candidates") != metadata.get("side_effect_candidates")
                    or handoff_risk.get("read_only_count") != len(metadata.get("read_only_candidates") or [])
                    or handoff_risk.get("personal_data_count") != len(metadata.get("personal_data_candidates") or [])
                    or handoff_risk.get("side_effect_count") != len(metadata.get("side_effect_candidates") or [])
                    or handoff_commands.get("boundary_contract") != f"integration boundary contract: {metadata.get('connector')}"
                    or handoff_commands.get("execution_matrix") != f"integration execution matrix: {metadata.get('connector')}"
                    or handoff_requirements.get("requires_boundary_contract") is not True
                    or handoff_requirements.get("requires_narrow_tool_schema") is not True
                    or handoff_requirements.get("requires_smoke_tests_before_nl_routing") is not True
                    or handoff_requirements.get("requires_audit_trail") is not True
                    or handoff_requirements.get("requires_pending_approval_for_blocked_risky_request") is not True
                    or handoff_requirements.get("requires_read_only_smoke") is not True
                    or handoff_requirements.get("requires_blocked_action_smoke") is not True
                    or handoff_requirements.get("requires_aggregate_smoke") is not True
                    or handoff_requirements.get("requires_fresh_route_review") is not True
                    or handoff_boundaries.get("natural_language_routing_enabled")
                    or handoff_boundaries.get("calls_external_service")
                    or handoff_boundaries.get("reads_personal_data")
                    or handoff_boundaries.get("executes_side_effect")
                    or handoff_boundaries.get("writes_memory")
                    or handoff_boundaries.get("authorizes_account_access")
                    or handoff_boundaries.get("authorizes_route_unlock")
                    or handoff_boundaries.get("authorizes_natural_language_routing")
                    or handoff_boundaries.get("authorizes_personal_data_read")
                    or handoff_boundaries.get("authorizes_side_effect")
                    or handoff_boundaries.get("authorizes_approval")
                    or handoff_boundaries.get("authorizes_model_call")
                    or handoff_boundaries.get("authorizes_tool_execution")
                    or handoff_boundaries.get("authorizes_external_service")
                    or handoff_boundaries.get("authorizes_execution")
                    or handoff_boundaries.get("authorizes_completion_claim")
                    or handoff_boundaries.get("approval_granted")
                    or handoff_boundaries.get("reusable_for_other_scope")
                    or handoff.get("next_step_requires_boundary_contract") is not True
                    or handoff.get("next_step_requires_action_preview") is not True
                    or handoff.get("next_route_lock_requires_fresh_review") is not True
                ):
                    raise SystemExit(f"Migration plan handoff diverged from flat metadata or safety contract: {metadata}")
            else:
                assert_contains(
                    result.response,
                    [
                        "Jarvis personal integration boundary report",
                        "Active integrations",
                        "Approval-gated integrations",
                        "Gated V2 personal connectors",
                        "calendar: active V2 connector",
                        "email: active V2 connector",
                        "messages: active V2 connectors",
                        "contacts: active V2 resolver",
                        "Still blocked legacy scope",
                        "find contact 가상연락처일",
                        "list calendars",
                        "search email from Alice",
                        "Safe migration rule",
                        "privacy report",
                        "safety status",
                    ],
                    case,
                )
                assert_read_only_metadata(result.tool_results[0].metadata, "Integration status")
                status_metadata = result.tool_results[0].metadata
                if any(fragment in result.response for fragment in ["/\x55sers/", "/private/", "/var/folders/", "/tmp/"]):
                    raise SystemExit(f"Integration status leaked local configured paths: {result.response}")
                if "no V2 connector is active yet" in result.response or "Not migrated yet" in result.response:
                    raise SystemExit(f"Integration status still reports obsolete connector state: {result.response}")
                if status_metadata.get("obsidian_vault_display") != "configured local vault" or status_metadata.get("obsidian_root_display") != "configured Jarvis root":
                    raise SystemExit(f"Integration status missed safe configured-path display metadata: {status_metadata}")
                expected_connector_names = ["calendar", "email", "messages", "contacts"]
                if status_metadata.get("gated_v2_connector_names") != expected_connector_names:
                    raise SystemExit(f"Integration status missed current V2 connector names: {status_metadata}")
                if status_metadata.get("gated_v2_connector_count") != len(expected_connector_names):
                    raise SystemExit(f"Integration status missed current V2 connector count: {status_metadata}")
                if status_metadata.get("unmigrated") != []:
                    raise SystemExit(f"Integration status should not mark current V2 connectors as unmigrated: {status_metadata}")
                if status_metadata.get("still_blocked_legacy_scope_count") != len(status_metadata.get("still_blocked_legacy_scope") or []):
                    raise SystemExit(f"Integration status legacy blocked scope count mismatch: {status_metadata}")

        long_action_result = runtime.registry.get("integration_action_preview").handler(
            {"connector": "email", "action": "search " + ("mailbox " * 80)}
        )
        print("[ok] direct bounded integration_action_preview")
        print(long_action_result.output[:600])
        print()
        if not long_action_result.ok:
            raise SystemExit("Long action preview should be safely bounded, not rejected.")
        assert_read_only_metadata(long_action_result.metadata, "Bounded long action preview")
        if len(long_action_result.metadata.get("action", "")) > 220:
            raise SystemExit("Long action was not bounded in metadata.")

        long_dry_run_result = runtime.registry.get("integration_dry_run_contract").handler(
            {
                "connector": "email",
                "action": "send " + ("draft reply " * 80),
                "target": "thread 123",
                "time_range": "today",
                "data_level": "draft-only",
                "verification": "confirm not sent",
            }
        )
        print("[ok] direct bounded integration_dry_run_contract")
        print(long_dry_run_result.output[:600])
        print()
        if not long_dry_run_result.ok:
            raise SystemExit("Long dry-run contract should be safely bounded, not rejected.")
        assert_read_only_metadata(long_dry_run_result.metadata, "Bounded dry-run contract")
        if len(long_dry_run_result.metadata.get("action", "")) > 220:
            raise SystemExit("Long dry-run action was not bounded in metadata.")

        long_runbook_result = runtime.registry.get("integration_runbook").handler(
            {
                "connector": "email",
                "action": "send " + ("draft reply " * 80),
                "target": "thread 123",
                "time_range": "today",
                "data_level": "draft-only",
                "verification": "confirm not sent",
                "rollback": "discard draft",
            }
        )
        print("[ok] direct bounded integration_runbook")
        print(long_runbook_result.output[:600])
        print()
        if not long_runbook_result.ok:
            raise SystemExit("Long integration runbook should be safely bounded, not rejected.")
        assert_read_only_metadata(long_runbook_result.metadata, "Bounded integration runbook")
        if len(long_runbook_result.metadata.get("action", "")) > 220:
            raise SystemExit("Long runbook action was not bounded in metadata.")

        long_promotion_result = runtime.registry.get("integration_promotion_gate").handler(
            {
                "connector": "email",
                "action": "send " + ("draft reply " * 80),
                "target": "thread 123",
                "time_range": "today",
                "data_level": "draft-only",
                "verification": "confirm not sent",
                "rollback": "discard draft",
                "tests": "blocked send smoke",
                "audit": "tool run receipt",
            }
        )
        print("[ok] direct bounded integration_promotion_gate")
        print(long_promotion_result.output[:600])
        print()
        if not long_promotion_result.ok:
            raise SystemExit("Long integration promotion gate should be safely bounded, not rejected.")
        assert_read_only_metadata(long_promotion_result.metadata, "Bounded integration promotion gate")
        if len(long_promotion_result.metadata.get("action", "")) > 220:
            raise SystemExit("Long promotion action was not bounded in metadata.")

        long_implementation_result = runtime.registry.get("integration_implementation_spec").handler(
            {
                "connector": "email",
                "action": "send " + ("draft reply " * 80),
                "target": "thread 123",
                "time_range": "today",
                "data_level": "draft-only",
                "verification": "confirm not sent",
                "rollback": "discard draft",
                "tests": "blocked send smoke",
                "audit": "tool run receipt",
            }
        )
        print("[ok] direct bounded integration_implementation_spec")
        print(long_implementation_result.output[:600])
        print()
        if not long_implementation_result.ok:
            raise SystemExit("Long integration implementation spec should be safely bounded, not rejected.")
        assert_read_only_metadata(long_implementation_result.metadata, "Bounded integration implementation spec")
        if len(long_implementation_result.metadata.get("action", "")) > 220:
            raise SystemExit("Long implementation spec action was not bounded in metadata.")

        long_preflight_result = runtime.registry.get("integration_preflight_contract").handler(
            {
                "connector": "email",
                "action": "send " + ("draft reply " * 80),
                "target": "thread 123",
                "time_range": "today",
                "data_level": "draft-only",
                "verification": "confirm not sent",
                "rollback": "discard draft",
                "tests": "blocked send smoke",
                "audit": "tool run receipt",
            }
        )
        print("[ok] direct bounded integration_preflight_contract")
        print(long_preflight_result.output[:600])
        print()
        if not long_preflight_result.ok:
            raise SystemExit("Long integration preflight contract should be safely bounded, not rejected.")
        assert_read_only_metadata(long_preflight_result.metadata, "Bounded integration preflight contract")
        if len(long_preflight_result.metadata.get("action", "")) > 220:
            raise SystemExit("Long preflight contract action was not bounded in metadata.")
        if long_preflight_result.metadata.get("preflight_state") != "PREFLIGHT_READY_FOR_REVIEW":
            raise SystemExit(f"Long preflight contract should be ready with full scope: {long_preflight_result.metadata}")

        local_path_cases = [
            "/\x55sers/example/Desktop/Claude code/private-mailbox",
            "/private/tmp/jarvis-secret-source",
            "/var/folders/zc/96ybqt1135x0x1_qmflb98t00000gn/T/jarvis-source",
            "/tmp/jarvis-secret-source",
        ]
        for raw_path in local_path_cases:
            local_scope_args = {
                "connector": "email",
                "action": "search mailbox metadata",
                "target": raw_path,
                "time_range": f"selected window from {raw_path}",
                "data_level": "metadata-only",
                "verification": f"fake rows only from {raw_path}",
                "rollback": f"no connector call at {raw_path}",
                "tests": f"blocked full body smoke for {raw_path}",
                "audit": f"metadata preview receipt for {raw_path}",
                "acceptance": f"acceptance gate passed for {raw_path}",
                "status": f"status api smoke passed for {raw_path}",
            }
            preflight_path_result = runtime.registry.get("integration_preflight_contract").handler(local_scope_args)
            proof_path_result = runtime.registry.get("integration_proof_bundle").handler(local_scope_args)
            route_path_result = runtime.registry.get("integration_route_lock").handler(local_scope_args)
            for label, result in [
                ("local-path integration preflight contract", preflight_path_result),
                ("local-path integration proof bundle", proof_path_result),
                ("local-path integration route lock", route_path_result),
            ]:
                print(f"[ok] direct {label}")
                print(result.output[:600])
                print()
                if not result.ok:
                    raise SystemExit(f"{label} should stay read-only and succeed with redacted scope.")
                assert_read_only_metadata(result.metadata, label)
                assert_local_paths_redacted(result, label)
            exact_args = preflight_path_result.metadata.get("exact_args") or {}
            if exact_args.get("target") != "<local-path>" or exact_args.get("time_range") != "selected window from <local-path>":
                raise SystemExit(f"Preflight exact_args should redact nested local scope metadata: {preflight_path_result.metadata}")

        reminder = runtime.handle("remind me to review Jarvis integrations")
        print(f"[blocked={not reminder.verified}] remind me to review Jarvis integrations")
        print(reminder.response[:1200])
        print()
        if reminder.verified:
            raise SystemExit("Expected reminder creation to require approval.")
        assert_contains(
            reminder.response,
            ["create_reminder", "explicit approval required", "Safety receipt", "pending approvals"],
            "reminder approval",
        )

        long_enablement_result = runtime.registry.get("integration_enablement_gate").handler(
            {
                "connector": "email",
                "action": "send " + ("draft reply " * 80),
                "target": "thread 123",
                "time_range": "today",
                "data_level": "draft-only",
                "verification": "confirm not sent",
                "rollback": "discard draft",
                "tests": "blocked send smoke",
                "audit": "tool run receipt",
                "acceptance": "acceptance gate passed with receipts",
            }
        )
        print("[ok] direct bounded integration_enablement_gate")
        print(long_enablement_result.output[:600])
        print()
        if not long_enablement_result.ok:
            raise SystemExit("Long integration enablement gate should be safely bounded, not rejected.")
        assert_read_only_metadata(long_enablement_result.metadata, "Bounded integration enablement gate")
        if len(long_enablement_result.metadata.get("action", "")) > 220:
            raise SystemExit("Long enablement gate action was not bounded in metadata.")
        if long_enablement_result.metadata.get("enablement_verdict") != "READY_FOR_APPROVAL_GATED_ENABLEMENT_REVIEW":
            raise SystemExit(f"Long enablement gate should be ready with full evidence: {long_enablement_result.metadata}")

        long_rehearsal_result = runtime.registry.get("integration_rehearsal_receipt").handler(
            {
                "connector": "email",
                "action": "send " + ("draft reply " * 80),
                "target": "thread 123",
                "time_range": "today",
                "data_level": "draft-only",
                "verification": "confirm not sent",
                "rollback": "discard draft",
                "tests": "blocked send smoke",
                "audit": "tool run receipt",
                "acceptance": "acceptance gate passed with receipts",
            }
        )
        print("[ok] direct bounded integration_rehearsal_receipt")
        print(long_rehearsal_result.output[:600])
        print()
        if not long_rehearsal_result.ok:
            raise SystemExit("Long integration rehearsal receipt should be safely bounded, not rejected.")
        assert_read_only_metadata(long_rehearsal_result.metadata, "Bounded integration rehearsal receipt")
        if len(long_rehearsal_result.metadata.get("action", "")) > 220:
            raise SystemExit("Long rehearsal receipt action was not bounded in metadata.")
        if long_rehearsal_result.metadata.get("rehearsal_state") != "REHEARSAL_READY_APPROVAL_GATED" or long_rehearsal_result.metadata.get("adapter_state") != "disabled":
            raise SystemExit(f"Long rehearsal receipt should be ready with disabled adapter: {long_rehearsal_result.metadata}")

        long_metadata_result = runtime.registry.get("integration_metadata_preview").handler(
            {
                "connector": "email",
                "action": "search " + ("mailbox metadata " * 80),
                "target": "inbox",
                "time_range": "last 7 days",
                "data_level": "metadata-only",
                "verification": "fake rows only",
                "tests": "blocked full body smoke",
                "audit": "metadata preview receipt",
                "acceptance": "acceptance gate passed",
            }
        )
        print("[ok] direct bounded integration_metadata_preview")
        print(long_metadata_result.output[:600])
        print()
        if not long_metadata_result.ok:
            raise SystemExit("Long integration metadata preview should be safely bounded, not rejected.")
        assert_read_only_metadata(long_metadata_result.metadata, "Bounded integration metadata preview")
        if len(long_metadata_result.metadata.get("action", "")) > 220:
            raise SystemExit("Long metadata preview action was not bounded in metadata.")
        if long_metadata_result.metadata.get("preview_state") != "METADATA_PREVIEW_READY" or long_metadata_result.metadata.get("adapter_state") != "metadata adapter disabled":
            raise SystemExit(f"Long metadata preview should be ready with disabled metadata adapter: {long_metadata_result.metadata}")

        route_scope_args = {
            "connector": "email",
            "action": "search mailbox metadata",
            "target": "inbox",
            "time_range": "last 7 days",
            "data_level": "metadata-only",
            "verification": "fake rows only",
            "rollback": "no connector call",
            "tests": "adapter acceptance passed",
            "audit": "adapter acceptance receipt",
            "acceptance": "acceptance gate passed",
            "status": "status api smoke passed",
        }
        review_result = runtime.registry.get("integration_implementation_review").handler(route_scope_args)
        expected_scope_hash = review_result.metadata.get("integration_scope_hash")
        if review_result.metadata.get("review_state") != "IMPLEMENTATION_REVIEW_READY" or len(str(expected_scope_hash or "")) != 16:
            raise SystemExit(f"Expected implementation review to produce a ready scope hash: {review_result.metadata}")

        matched_route_result = runtime.registry.get("integration_route_lock").handler(
            {**route_scope_args, "expected_scope_hash": expected_scope_hash}
        )
        print("[ok] direct integration_route_lock expected hash match")
        print(matched_route_result.output[:600])
        print()
        assert_read_only_metadata(matched_route_result.metadata, "Expected-hash integration route lock")
        if (
            matched_route_result.metadata.get("route_lock_state") != "ROUTE_LOCK_READY_FOR_EXPLICIT_REVIEW"
            or matched_route_result.metadata.get("route_unlock_candidate") is not True
            or matched_route_result.metadata.get("expected_scope_hash_required") is not True
            or matched_route_result.metadata.get("scope_hash_matches_expected") is not True
            or matched_route_result.metadata.get("missing_fields")
        ):
            raise SystemExit(f"Expected route lock to accept matching expected scope hash: {matched_route_result.metadata}")

        mismatched_route_result = runtime.registry.get("integration_route_lock").handler(
            {**route_scope_args, "expected_scope_hash": "deadbeefdeadbeef"}
        )
        print("[ok] direct integration_route_lock expected hash mismatch")
        print(mismatched_route_result.output[:600])
        print()
        assert_read_only_metadata(mismatched_route_result.metadata, "Mismatched-hash integration route lock")
        if (
            mismatched_route_result.metadata.get("route_lock_state") != "ROUTE_LOCK_HELD"
            or mismatched_route_result.metadata.get("route_unlock_candidate")
            or mismatched_route_result.metadata.get("scope_hash_matches_expected")
            or mismatched_route_result.metadata.get("expected_scope_hash_required") is not True
            or "expected scope hash match" not in mismatched_route_result.metadata.get("missing_fields", [])
        ):
            raise SystemExit(f"Expected route lock to reject mismatched expected scope hash: {mismatched_route_result.metadata}")

        tools = runtime.handle("list tools personal")
        print("[ok] list tools personal")
        print(tools.response[:1200])
        print()
        assert_contains(tools.response, ["integration_action_preview", "integration_boundary_contract", "integration_readiness_report", "legacy_connector_migration_audit", "integration_scope_packet", "integration_dry_run_contract", "integration_runbook", "integration_promotion_gate", "integration_implementation_spec", "integration_preflight_contract", "integration_enablement_gate", "integration_rehearsal_receipt", "integration_metadata_preview", "integration_proof_bundle", "integration_implementation_review", "integration_route_lock", "integration_execution_matrix", "integration_adapter_manifest", "integration_adapter_probe", "integration_adapter_acceptance"], "list tools personal")

        class FakeMacFailure:
            returncode = 1
            stdout = ""
            stderr = "macOS denied /\x55sers/example/private/personal"

        original_run = personal_tools.subprocess.run
        try:
            personal_tools.subprocess.run = lambda *_args, **_kwargs: FakeMacFailure()  # type: ignore[assignment]
            failed_vault = runtime.registry.get("open_jarvis_vault").handler({})
            failed_reminder = runtime.registry.get("create_reminder").handler({"title": "check smoke"})
        finally:
            personal_tools.subprocess.run = original_run  # type: ignore[assignment]
        if failed_vault.ok or "Could not open Jarvis vault." not in failed_vault.output:
            raise SystemExit(f"open_jarvis_vault failure should return friendly output: {failed_vault.output}")
        for expected in ["JARVIS_OBSIDIAN_VAULT", "Finder > Get Info > Sharing & Permissions", "setup check", "then retry"]:
            if expected not in failed_vault.output:
                raise SystemExit(f"open_jarvis_vault failure missed actionable guidance {expected}: {failed_vault.output}")
        if "/\x55sers/operator" in failed_vault.output or "macOS denied" in failed_vault.output:
            raise SystemExit(f"open_jarvis_vault failure leaked raw stderr: {failed_vault.output}")
        if failed_vault.metadata.get("returncode") != 1 or failed_vault.metadata.get("controls_computer") is not True:
            raise SystemExit(f"open_jarvis_vault failure missed diagnostic metadata: {failed_vault.metadata}")
        assert_reminder_outcome_unknown(
            failed_reminder,
            "create_reminder nonzero exit",
        )
        if "/\x55sers/operator" in failed_reminder.output or "macOS denied" in failed_reminder.output:
            raise SystemExit(f"create_reminder failure leaked raw stderr: {failed_reminder.output}")
        if failed_reminder.metadata.get("returncode") != 1 or failed_reminder.metadata.get("external_side_effect") is not True:
            raise SystemExit(f"create_reminder failure missed diagnostic metadata: {failed_reminder.metadata}")

        class FakeMacSuccess:
            returncode = 0
            stdout = ""
            stderr = ""

        try:
            personal_tools.subprocess.run = lambda *_args, **_kwargs: FakeMacSuccess()  # type: ignore[assignment]
            opened_vault = runtime.registry.get("open_jarvis_vault").handler({})
        finally:
            personal_tools.subprocess.run = original_run  # type: ignore[assignment]
        if not opened_vault.ok or "Opened Jarvis vault." not in opened_vault.output:
            raise SystemExit(f"open_jarvis_vault success should use a safe display receipt: {opened_vault.output}")
        if any(fragment in opened_vault.output for fragment in ["/\x55sers/", "/private/", "/var/folders/", "/tmp/"]):
            raise SystemExit(f"open_jarvis_vault success leaked local path output: {opened_vault.output}")
        if opened_vault.metadata.get("path_display") != "Jarvis vault" or not opened_vault.metadata.get("path"):
            raise SystemExit(f"open_jarvis_vault success should preserve exact path plus safe display metadata: {opened_vault.metadata}")

        try:
            def raise_if_called(*_args, **_kwargs):
                raise AssertionError("create_reminder should reject path-shaped titles before AppleScript")

            personal_tools.subprocess.run = raise_if_called  # type: ignore[assignment]
            path_reminders = [
                runtime.registry.get("create_reminder").handler({"title": sample})
                for sample in [
                    "/\x55sers/example/private/reminder-title",
                    "/private/tmp/jarvis-reminder-title",
                    "/var/folders/zc/jarvis-reminder-title",
                    "/tmp/jarvis-reminder-title",
                ]
            ]
        finally:
            personal_tools.subprocess.run = original_run  # type: ignore[assignment]
        for path_reminder in path_reminders:
            if path_reminder.ok or path_reminder.metadata.get("reason") != "invalid_title":
                raise SystemExit(f"create_reminder should reject path-shaped titles: {path_reminder.metadata}")
            if (
                path_reminder.metadata.get("title") != "<local-path>"
                or path_reminder.metadata.get("executes_tools")
                or path_reminder.metadata.get("external_side_effect")
                or path_reminder.metadata.get("controls_computer")
            ):
                raise SystemExit(f"create_reminder path refusal should be local and redacted: {path_reminder.metadata}")
            if any(fragment in path_reminder.output for fragment in ["/\x55sers/", "/private/", "/var/folders/", "/tmp/"]):
                raise SystemExit(f"create_reminder path refusal leaked raw title: {path_reminder.output}")
            if any(fragment in str(path_reminder.metadata) for fragment in ["/\x55sers/", "/private/", "/var/folders/", "/tmp/"]):
                raise SystemExit(f"create_reminder path refusal leaked raw metadata: {path_reminder.metadata}")

        try:
            def raise_mac_failure(*_args, **_kwargs):
                raise RuntimeError("macOS crashed near /\x55sers/example/private/personal")

            personal_tools.subprocess.run = raise_mac_failure  # type: ignore[assignment]
            exception_vault = runtime.registry.get("open_jarvis_vault").handler({})
            exception_reminder = runtime.registry.get("create_reminder").handler({"title": "check smoke"})
        finally:
            personal_tools.subprocess.run = original_run  # type: ignore[assignment]
        if exception_vault.ok or "Could not open Jarvis vault." not in exception_vault.output:
            raise SystemExit(f"open_jarvis_vault exception should return friendly output: {exception_vault.output}")
        for expected in ["JARVIS_OBSIDIAN_VAULT", "Finder > Get Info > Sharing & Permissions", "setup check", "then retry"]:
            if expected not in exception_vault.output:
                raise SystemExit(f"open_jarvis_vault exception missed actionable guidance {expected}: {exception_vault.output}")
        if "/\x55sers/operator" in exception_vault.output or "crashed near" in exception_vault.output:
            raise SystemExit(f"open_jarvis_vault exception leaked raw text: {exception_vault.output}")
        if exception_vault.metadata.get("exception_type") != "RuntimeError":
            raise SystemExit(f"open_jarvis_vault exception missed diagnostic metadata: {exception_vault.metadata}")
        assert_reminder_outcome_unknown(
            exception_reminder,
            "create_reminder subprocess exception",
        )
        if "/\x55sers/operator" in exception_reminder.output or "crashed near" in exception_reminder.output:
            raise SystemExit(f"create_reminder exception leaked raw text: {exception_reminder.output}")
        if exception_reminder.metadata.get("exception_type") != "RuntimeError":
            raise SystemExit(f"create_reminder exception missed diagnostic metadata: {exception_reminder.metadata}")


if __name__ == "__main__":
    main()
