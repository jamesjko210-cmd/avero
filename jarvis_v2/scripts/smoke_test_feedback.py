from __future__ import annotations

import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory

from jarvis_v2.agent.planner import RuleBasedPlanner
from jarvis_v2.memory.store import MemoryRecord
from jarvis_v2.scripts.test_runtime import make_temp_runtime
from jarvis_v2.tools.feedback import (
    _feedback_handoff_metadata,
    _feedback_metadata_bool,
    _failure_patch_handoff_ready,
    _failure_learning_closure_token_boundary_ready,
    _failure_learning_closure_token_sha256,
    _failure_learning_record_token_ready,
    _failure_learning_record_token_sha256,
    _failure_patch_review_token_sha256,
    _failure_regression_test_contract_sha256,
    make_feedback_tools,
)


FEEDBACK_TOOL_NAMES = {
    "record_feedback",
    "feedback_report",
    "save_feedback_report",
    "feedback_actions",
    "save_feedback_actions",
    "failure_to_test_preview",
    "repeated_failure_clusters",
    "failure_promotion_packet",
    "failure_implementation_packet",
    "failure_apply_contract",
    "failure_learning_cockpit",
    "failure_patch_receipt_packet",
    "failure_patch_application_bridge",
    "failure_patch_completion_gate",
    "failure_patch_handoff_packet",
    "failure_patch_closeout_packet",
    "failure_learning_record_packet",
    "failure_learning_closure_ledger",
}


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


def assert_planner_routes_feedback_phone_aliases() -> None:
    planner = RuleBasedPlanner()
    for text in (
        "feedback please",
        "feedback report please",
        "show feedback",
        "show latest feedback",
        "latest feedback",
    ):
        plan = planner.plan(text)
        if [(action.tool_name, action.args) for action in plan.actions] != [("feedback_report", {})]:
            raise SystemExit(f"planner missed feedback report alias {text!r}: {plan.actions}")
    for text in ("feedback actions please", "jarvis feedback actions", "feedback improvements please"):
        plan = planner.plan(text)
        if [(action.tool_name, action.args) for action in plan.actions] != [("feedback_actions", {})]:
            raise SystemExit(f"planner missed feedback actions alias {text!r}: {plan.actions}")
    record_plan = planner.plan("feedback: Jarvis should explain risky actions clearly")
    if [(action.tool_name, action.args) for action in record_plan.actions] != [
        ("record_feedback", {"body": "Jarvis should explain risky actions clearly"})
    ]:
        raise SystemExit(f"planner should preserve explicit feedback capture: {record_plan.actions}")


def assert_contains(text: str, expected: list[str], label: str) -> None:
    missing = [item for item in expected if item not in text]
    if missing:
        raise SystemExit(f"{label} missing expected text: {missing}")


def text_sha256(value: str) -> str:
    text = " ".join(value.strip().split())
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def assert_feedback_metadata_bool_is_exact() -> None:
    if _feedback_metadata_bool(True) is not True:
        raise SystemExit("feedback exact metadata bool rejected True")
    if _feedback_metadata_bool(False) is not False:
        raise SystemExit("feedback exact metadata bool rejected False")
    for value in ("true", "false", "yes", "no", 1, 0, [True], {"ready": True}, None):
        if _feedback_metadata_bool(value) is not False:
            raise SystemExit(f"feedback exact metadata bool accepted malformed truthy value: {value!r}")
    if _feedback_metadata_bool("false", default=True) is not True:
        raise SystemExit("feedback exact metadata bool did not preserve explicit default")


def assert_feedback_malformed_handoff_flags_are_exact() -> None:
    handoff = {
        "source": "feedback_report",
        "state_changed": "false",
        "changed": [],
        "content_in_handoff": "true",
        "next_commands": ["feedback report"],
        "boundaries": {"read_only": True},
    }
    metadata = _feedback_handoff_metadata("feedback_report_handoff", handoff)
    for key in ["state_changed", "feedback_report_state_changed"]:
        if metadata.get(key) is not False:
            raise SystemExit(f"feedback handoff accepted malformed state_changed for {key}: {metadata}")
    for key in ["content_in_handoff", "feedback_report_content_in_handoff"]:
        if metadata.get(key) is not False:
            raise SystemExit(f"feedback handoff accepted malformed content_in_handoff for {key}: {metadata}")


def assert_vault_relative_receipt(result, root: Path, prefix: str, label: str) -> None:
    metadata = result.tool_results[0].metadata if hasattr(result, "tool_results") else result.metadata
    response = result.response if hasattr(result, "response") else result.output
    receipt_line = response.split("\n", 1)[0]
    path_text = str(metadata.get("path") or "")
    path_display = metadata.get("path_display")
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


def assert_no_local_path(value: str, label: str) -> None:
    for fragment in ["/\x55sers/", "/private/", "/var/folders/", "/tmp/"]:
        if fragment in value:
            raise SystemExit(f"{label} leaked a local path: {value}")


def assert_feedback_no_future_authority(metadata: dict, label: str) -> None:
    for key in ["authorizes_execution", "authorizes_completion_claim", "approval_granted"]:
        if metadata.get(key) is not False:
            raise SystemExit(f"{label} should keep {key}=False: {metadata}")


def assert_basic_feedback_handoff(
    metadata: dict,
    handoff_key: str,
    label: str,
    *,
    state_changed: bool,
    changed: list[str],
    content_in_handoff: bool,
    writes_files: bool,
    writes_memory: bool,
    writes_notes: bool,
    writes_database: bool,
) -> None:
    handoff = metadata.get(handoff_key)
    if not isinstance(handoff, dict) or metadata.get(f"{handoff_key}_ready") is not True:
        raise SystemExit(f"{label} missed {handoff_key}: {metadata}")
    prefix = handoff_key.removesuffix("_handoff")
    raw_next = handoff.get("next_commands") or []
    if isinstance(raw_next, dict):
        expected_next = [str(value) for value in raw_next.values() if str(value or "").strip()]
    else:
        expected_next = [str(value) for value in raw_next if str(value or "").strip()]
    expected_first = expected_next[0] if expected_next else ""
    for key, expected in (
        ("handoff_ready", True),
        ("ready_for_operator", True),
        ("state_changed", state_changed),
        ("changed", changed),
        ("content_in_handoff", content_in_handoff),
        ("authorizes_execution", False),
        ("authorizes_completion_claim", False),
        ("approval_granted", False),
    ):
        if handoff.get(key) != expected:
            raise SystemExit(f"{label} handoff missed {key}={expected}: {handoff}")
        if key != "handoff_ready" and metadata.get(key) != expected:
            raise SystemExit(f"{label} metadata missed {key}={expected}: {metadata}")
    for key, expected in (
        (f"{prefix}_handoff_ready", True),
        (f"{prefix}_ready_for_operator", True),
        (f"{prefix}_state_changed", state_changed),
        (f"{prefix}_changed", changed),
        (f"{prefix}_content_in_handoff", content_in_handoff),
        (f"{prefix}_authorizes_execution", False),
        (f"{prefix}_authorizes_completion_claim", False),
        (f"{prefix}_approval_granted", False),
        (f"{prefix}_next_safe_command", expected_first),
        (f"{prefix}_next_safe_commands", expected_next),
        (f"{prefix}_next_safe_command_count", len(expected_next)),
    ):
        if metadata.get(key) != expected:
            raise SystemExit(f"{label} prefixed alias {key} mismatch: {metadata}")
    for container, container_label in ((handoff, "handoff"), (metadata, "metadata")):
        if container.get("next_safe_command") != expected_first:
            raise SystemExit(f"{label} {container_label} next_safe_command mismatch: {container}")
        if container.get("next_safe_commands") != expected_next:
            raise SystemExit(f"{label} {container_label} next_safe_commands mismatch: {container}")
        if container.get("next_safe_command_count") != len(expected_next):
            raise SystemExit(f"{label} {container_label} next_safe_command_count mismatch: {container}")
    boundaries = handoff.get("boundaries") or {}
    expected_boundaries = {
        "writes_files": writes_files,
        "writes_memory": writes_memory,
        "writes_notes": writes_notes,
        "writes_database": writes_database,
        "queues_approval": False,
        "authorizes_execution": False,
        "authorizes_completion_claim": False,
        "approval_granted": False,
        "controls_computer": False,
        "external_side_effect": False,
    }
    for key, expected in expected_boundaries.items():
        if boundaries.get(key) is not expected:
            raise SystemExit(f"{label} boundary {key} should be {expected}: {handoff}")
    for key, expected in (
        ("writes_files", writes_files),
        ("writes_memory", writes_memory),
        ("writes_notes", writes_notes),
        ("writes_database", writes_database),
    ):
        if metadata.get(key) is not expected:
            raise SystemExit(f"{label} flat {key} should be {expected}: {metadata}")


def assert_feedback_result_no_future_authority(result, label: str) -> None:
    tool_results = getattr(result, "tool_results", None)
    if tool_results is None:
        assert_feedback_no_future_authority(result.metadata, label)
        return
    for tool_result in tool_results:
        if tool_result.tool_name in FEEDBACK_TOOL_NAMES:
            assert_feedback_no_future_authority(tool_result.metadata, f"{label} / {tool_result.tool_name}")


def assert_patch_review_contract(metadata: dict, label: str) -> None:
    rows = metadata.get("patch_review_contract_rows") or []
    token = str(metadata.get("failure_patch_review_token_sha256") or "")
    expected_items = {
        "review_scope_binding",
        "applied_patch_bridge_only",
        "verification_receipt_boundary",
        "completion_claim_boundary",
        "learning_record_boundary",
        "fresh_patch_review_token",
    }
    if metadata.get("patch_review_contract_ready") is not True:
        raise SystemExit(f"{label} missed ready patch review contract: {metadata}")
    if metadata.get("patch_review_contract_row_count") != len(rows) or len(rows) != 6:
        raise SystemExit(f"{label} missed patch review contract rows: {metadata}")
    if {row.get("item") for row in rows} != expected_items:
        raise SystemExit(f"{label} missed expected patch review contract items: {metadata}")
    if metadata.get("failure_patch_review_token_present") is not True or len(token) != 64:
        raise SystemExit(f"{label} missed patch review token: {metadata}")
    recomputed = _failure_patch_review_token_sha256(
        target_test=str(metadata.get("target_test") or ""),
        target_file=str(metadata.get("target_file") or ""),
        apply_contract_sha256=str(metadata.get("apply_contract_sha256") or ""),
        failure_regression_test_contract_sha256=str(metadata.get("failure_regression_test_contract_sha256") or ""),
        patch_receipt_sha256=str(metadata.get("patch_receipt_sha256") or ""),
        review_contract_rows=list(rows),
    )
    if recomputed != token:
        raise SystemExit(f"{label} patch review token should be reproducible: {metadata}")
    tampered = _failure_patch_review_token_sha256(
        target_test=str(metadata.get("target_test") or ""),
        target_file=str(metadata.get("target_file") or ""),
        apply_contract_sha256=str(metadata.get("apply_contract_sha256") or ""),
        failure_regression_test_contract_sha256=str(metadata.get("failure_regression_test_contract_sha256") or ""),
        patch_receipt_sha256=str(metadata.get("patch_receipt_sha256") or ""),
        review_contract_rows=list(rows),
        authorizes_patch_application=True,
    )
    if tampered == token:
        raise SystemExit(f"{label} patch review token should bind patch-application authority: {metadata}")
    tampered_rows = [dict(row) for row in rows]
    tampered_rows[0]["authorizes_approval"] = True
    tampered_approval_row = _failure_patch_review_token_sha256(
        target_test=str(metadata.get("target_test") or ""),
        target_file=str(metadata.get("target_file") or ""),
        apply_contract_sha256=str(metadata.get("apply_contract_sha256") or ""),
        failure_regression_test_contract_sha256=str(metadata.get("failure_regression_test_contract_sha256") or ""),
        patch_receipt_sha256=str(metadata.get("patch_receipt_sha256") or ""),
        review_contract_rows=tampered_rows,
    )
    if tampered_approval_row == token:
        raise SystemExit(f"{label} patch review token should bind row approval authority: {metadata}")
    tampered_exact_test_contract = _failure_patch_review_token_sha256(
        target_test=str(metadata.get("target_test") or ""),
        target_file=str(metadata.get("target_file") or ""),
        apply_contract_sha256=str(metadata.get("apply_contract_sha256") or ""),
        failure_regression_test_contract_sha256="0" * 64,
        patch_receipt_sha256=str(metadata.get("patch_receipt_sha256") or ""),
        review_contract_rows=list(rows),
    )
    if tampered_exact_test_contract == token:
        raise SystemExit(f"{label} patch review token should bind exact regression-test contract hash: {metadata}")
    for flag in [
        "review_authorizes_patch_application",
        "review_authorizes_file_write",
        "review_authorizes_tool_execution",
        "review_authorizes_model_call",
        "review_authorizes_personal_data_read",
        "review_authorizes_external_side_effect",
        "review_authorizes_approval",
        "review_authorizes_completion_claim",
        "review_authorizes_learning_record",
        "review_reusable_for_next_patch",
        "review_token_authorizes_patch_application",
        "review_token_authorizes_file_write",
        "review_token_authorizes_tool_execution",
        "review_token_authorizes_model_call",
        "review_token_authorizes_personal_data_read",
        "review_token_authorizes_external_side_effect",
        "review_token_authorizes_approval",
        "review_token_authorizes_completion_claim",
        "review_token_authorizes_learning_record",
        "review_token_reusable_for_next_patch",
    ]:
        if metadata.get(flag) is not False:
            raise SystemExit(f"{label} should keep {flag}=False: {metadata}")
    if metadata.get("next_patch_requires_fresh_review_token") is not True:
        raise SystemExit(f"{label} missed fresh next patch review token requirement: {metadata}")
    if metadata.get("review_token_binds_exact_test_contract") is not True:
        raise SystemExit(f"{label} missed exact regression-test contract binding flag: {metadata}")
    for row in rows:
        for row_flag in [
            "authorizes_patch_application",
            "authorizes_file_write",
            "authorizes_tool_execution",
            "authorizes_model_call",
            "authorizes_personal_data_read",
            "authorizes_external_side_effect",
            "authorizes_approval",
            "authorizes_completion_claim",
            "authorizes_learning_record",
            "reusable_for_next_patch",
        ]:
            if row.get(row_flag) is not False:
                raise SystemExit(f"{label} row should keep {row_flag}=False: {metadata}")


def assert_exact_test_contract(metadata: dict, label: str) -> None:
    token = str(metadata.get("failure_regression_test_contract_sha256") or "")
    if metadata.get("failure_regression_test_contract_present") is not True or len(token) != 64:
        raise SystemExit(f"{label} missed exact regression test contract: {metadata}")
    if metadata.get("failure_regression_test_contract_ready") is not True or metadata.get("exact_test_contract_ready") is not True:
        raise SystemExit(f"{label} missed ready exact test contract: {metadata}")
    if metadata.get("exact_test_contract_requires_pre_patch_failure") is not True:
        raise SystemExit(f"{label} missed pre-patch failure requirement: {metadata}")
    for flag in [
        "exact_test_contract_authorizes_file_write",
        "exact_test_contract_authorizes_patch_application",
        "exact_test_contract_authorizes_tool_execution",
        "exact_test_contract_authorizes_completion_claim",
        "exact_test_contract_reusable_for_next_patch",
    ]:
        if metadata.get(flag) is not False:
            raise SystemExit(f"{label} should keep {flag}=False: {metadata}")
    if metadata.get("observed_failures") is not None:
        recomputed = _failure_regression_test_contract_sha256(
            surface=str(metadata.get("selected_surface") or ""),
            target_test=str(metadata.get("target_test") or ""),
            target_file=str(metadata.get("target_file") or ""),
            focused_command=str(metadata.get("focused_command") or ""),
            expected_behavior=str(metadata.get("expected_behavior") or ""),
            observed_failures=list(metadata.get("observed_failures") or []),
            rollback_note=str(metadata.get("rollback_note") or ""),
            assertions=[],
        )
        if recomputed == token:
            raise SystemExit(f"{label} exact test contract should also bind assertion list, not only headings: {metadata}")


def assert_command_first_proof_queue(
    result,
    label: str,
    *,
    queue_key: str = "proof_queue",
    proof_key: str = "next_proof_command",
) -> None:
    metadata = result.metadata
    output = result.output
    queue = metadata.get(queue_key) or []
    if not queue:
        raise SystemExit(f"{label} missed proof queue metadata: {metadata}")
    if "- next proof command:" in output:
        raise SystemExit(f"{label} should not expose proof-command wording to operators: {output}")
    required = str(metadata.get("next_required_command") or "")
    if not required:
        raise SystemExit(f"{label} missed next_required_command alias: {metadata}")
    if f"- next required command: `{required}`" not in output:
        raise SystemExit(f"{label} output missed next required command {required!r}: {output}")
    if metadata.get(proof_key) != queue[0]:
        raise SystemExit(f"{label} should preserve proof queue head metadata: {metadata}")
    if metadata.get("next_command") and metadata.get("next_required_command") != metadata.get("next_command"):
        raise SystemExit(f"{label} next_required_command should mirror next_command: {metadata}")


def assert_failure_learning_record_token(metadata: dict, label: str) -> None:
    token = str(metadata.get("failure_learning_record_token_sha256") or "")
    if metadata.get("failure_learning_record_token_present") is not True or len(token) != 64:
        raise SystemExit(f"{label} missed failure learning record token: {metadata}")
    if not _failure_learning_record_token_ready(metadata):
        raise SystemExit(f"{label} failure learning record token failed production validator: {metadata}")
    for flag in [
        "record_token_authorizes_learning_record",
        "record_token_authorizes_memory_write",
        "record_token_authorizes_file_write",
        "record_token_authorizes_tool_execution",
        "record_token_authorizes_model_call",
        "record_token_authorizes_personal_data_read",
        "record_token_authorizes_external_side_effect",
        "record_token_authorizes_approval",
        "record_token_authorizes_completion_claim",
        "record_token_reusable_for_next_learning_record",
    ]:
        if metadata.get(flag) is not False:
            raise SystemExit(f"{label} should keep {flag}=False: {metadata}")
    if metadata.get("next_learning_record_requires_fresh_record_token") is not True:
        raise SystemExit(f"{label} missed fresh next learning record token requirement: {metadata}")
    if metadata.get("patch_review_contract_rows") is not None:
        recomputed = _failure_learning_record_token_sha256(
            target_test=str(metadata.get("target_test") or ""),
            target_file=str(metadata.get("target_file") or ""),
            after_action_learning=str(metadata.get("after_action_learning") or ""),
            record_target=str(metadata.get("learning_record_target") or ""),
            regression_link=str(metadata.get("regression_link") or ""),
            durability_note=str(metadata.get("durability_note") or ""),
            patch_receipt_sha256=str(metadata.get("patch_receipt_sha256") or ""),
            learning_record_sha256=str(metadata.get("learning_record_sha256") or ""),
            apply_contract_sha256=str(metadata.get("apply_contract_sha256") or ""),
            review_contract_rows=list(metadata.get("patch_review_contract_rows") or []),
        )
        if recomputed != token:
            raise SystemExit(f"{label} learning record token should be reproducible: {metadata}")
        tampered = _failure_learning_record_token_sha256(
            target_test=str(metadata.get("target_test") or ""),
            target_file=str(metadata.get("target_file") or ""),
            after_action_learning=str(metadata.get("after_action_learning") or ""),
            record_target=str(metadata.get("learning_record_target") or ""),
            regression_link=str(metadata.get("regression_link") or ""),
            durability_note=str(metadata.get("durability_note") or ""),
            patch_receipt_sha256=str(metadata.get("patch_receipt_sha256") or ""),
            learning_record_sha256=str(metadata.get("learning_record_sha256") or ""),
            apply_contract_sha256=str(metadata.get("apply_contract_sha256") or ""),
            review_contract_rows=list(metadata.get("patch_review_contract_rows") or []),
            authorizes_learning_record=True,
        )
        if tampered == token:
            raise SystemExit(f"{label} learning record token should bind learning-record authority: {metadata}")
        tampered_metadata = dict(metadata)
        tampered_metadata["record_token_authorizes_learning_record"] = True
        if _failure_learning_record_token_ready(tampered_metadata):
            raise SystemExit(f"{label} learning record token validator should reject authority tampering: {metadata}")
        tampered_rows = [dict(row) for row in (metadata.get("patch_review_contract_rows") or [])]
        if tampered_rows:
            tampered_rows[0]["authorizes_file_write"] = True
            tampered_row_metadata = dict(metadata)
            tampered_row_metadata["patch_review_contract_rows"] = tampered_rows
            if _failure_learning_record_token_ready(tampered_row_metadata):
                raise SystemExit(f"{label} learning record token validator should reject review-row tampering: {metadata}")
            tampered_file_write = _failure_learning_record_token_sha256(
                target_test=str(metadata.get("target_test") or ""),
                target_file=str(metadata.get("target_file") or ""),
                after_action_learning=str(metadata.get("after_action_learning") or ""),
                record_target=str(metadata.get("learning_record_target") or ""),
                regression_link=str(metadata.get("regression_link") or ""),
                durability_note=str(metadata.get("durability_note") or ""),
                patch_receipt_sha256=str(metadata.get("patch_receipt_sha256") or ""),
                learning_record_sha256=str(metadata.get("learning_record_sha256") or ""),
                apply_contract_sha256=str(metadata.get("apply_contract_sha256") or ""),
                review_contract_rows=tampered_rows,
            )
            if tampered_file_write == token:
                raise SystemExit(f"{label} learning record token should bind review-row file-write authority: {metadata}")


def assert_failure_learning_closure_token_boundary(metadata: dict, label: str) -> None:
    token = str(metadata.get("failure_learning_closure_token_sha256") or "")
    rows = metadata.get("failure_learning_closure_token_boundary_rows") or []
    expected_items = {
        "failure_learning_closure_token",
        "record_token_carry_forward",
        "fresh_patch_learning_review",
    }
    if metadata.get("failure_learning_closure_token_present") is not True or len(token) != 64:
        raise SystemExit(f"{label} missed failure learning closure token: {metadata}")
    if metadata.get("failure_learning_closure_token_boundary_row_count") != len(rows) or len(rows) != 3:
        raise SystemExit(f"{label} missed closure token boundary rows: {metadata}")
    if metadata.get("failure_learning_closure_token_boundary_ready") is not True:
        raise SystemExit(f"{label} missed closure token boundary ready flag: {metadata}")
    if not _failure_learning_closure_token_boundary_ready(
        rows,
        token_sha256=token,
        record_token_sha256=str(metadata.get("failure_learning_record_token_sha256") or ""),
    ):
        raise SystemExit(f"{label} closure token boundary failed production validator: {metadata}")
    if metadata.get("closure_stage_row_count") != len(metadata.get("closure_stage_rows") or []):
        raise SystemExit(f"{label} missed closure stage row count: {metadata}")
    if len(metadata.get("closure_stage_rows") or []) != 7:
        raise SystemExit(f"{label} missed closure stage rows: {metadata}")
    for row in metadata.get("closure_stage_rows") or []:
        for row_flag in [
            "authorizes_patch_application",
            "authorizes_file_write",
            "authorizes_tool_execution",
            "authorizes_model_call",
            "authorizes_personal_data_read",
            "authorizes_external_side_effect",
            "authorizes_approval",
            "authorizes_completion_claim",
            "authorizes_learning_record",
            "reusable_for_next_patch",
            "reusable_for_next_learning_record",
            "reusable_for_next_completion_claim",
        ]:
            if row.get(row_flag) is not False:
                raise SystemExit(f"{label} closure stage row should keep {row_flag}=False: {metadata}")
    if {row.get("item") for row in rows} != expected_items:
        raise SystemExit(f"{label} missed closure token boundary items: {metadata}")
    for row in rows:
        if row.get("source") != "failure_learning_closure_ledger" or row.get("token_sha256") != token:
            raise SystemExit(f"{label} closure boundary row missed source/hash binding: {metadata}")
        for row_flag in [
            "authorizes_patch_application",
            "authorizes_file_write",
            "authorizes_tool_execution",
            "authorizes_model_call",
            "authorizes_personal_data_read",
            "authorizes_external_side_effect",
            "authorizes_approval",
            "authorizes_completion_claim",
            "authorizes_learning_record",
            "reusable_for_next_patch",
            "reusable_for_next_learning_record",
            "reusable_for_next_completion_claim",
        ]:
            if row.get(row_flag) is not False:
                raise SystemExit(f"{label} closure boundary row should keep {row_flag}=False: {metadata}")
    tampered_rows = [dict(row) for row in rows]
    tampered_rows[0]["authorizes_completion_claim"] = True
    if _failure_learning_closure_token_boundary_ready(
        tampered_rows,
        token_sha256=token,
        record_token_sha256=str(metadata.get("failure_learning_record_token_sha256") or ""),
    ):
        raise SystemExit(f"{label} closure token boundary validator should reject authority tampering: {metadata}")
    tampered_status_rows = [dict(row) for row in rows]
    tampered_status_rows[2]["status"] = "fresh_review_optional"
    if _failure_learning_closure_token_boundary_ready(
        tampered_status_rows,
        token_sha256=token,
        record_token_sha256=str(metadata.get("failure_learning_record_token_sha256") or ""),
    ):
        raise SystemExit(f"{label} closure token boundary validator should reject status tampering: {metadata}")
    for flag in [
        "closure_token_reusable_for_next_patch",
        "closure_token_reusable_for_next_learning_record",
        "closure_token_reusable_for_next_completion_claim",
        "closure_token_authorizes_patch_application",
        "closure_token_authorizes_file_write",
        "closure_token_authorizes_tool_execution",
        "closure_token_authorizes_model_call",
        "closure_token_authorizes_personal_data_read",
        "closure_token_authorizes_external_side_effect",
        "closure_token_authorizes_approval",
        "closure_token_authorizes_completion_claim",
        "closure_token_authorizes_learning_record",
    ]:
        if metadata.get(flag) is not False:
            raise SystemExit(f"{label} should keep {flag}=False: {metadata}")
    if metadata.get("patch_review_contract_rows") is not None:
        recomputed = _failure_learning_closure_token_sha256(
            target_test=str(metadata.get("target_test") or ""),
            target_file=str(metadata.get("target_file") or ""),
            patch_receipt_sha256=str(metadata.get("patch_receipt_sha256") or ""),
            learning_record_sha256=str(metadata.get("learning_record_sha256") or ""),
            apply_contract_sha256=str(metadata.get("apply_contract_sha256") or ""),
            failure_patch_review_token_sha256=str(metadata.get("failure_patch_review_token_sha256") or ""),
            failure_learning_record_token_sha256=str(metadata.get("failure_learning_record_token_sha256") or ""),
            after_action_learning=str(metadata.get("after_action_learning") or ""),
            learning_record_target=str(metadata.get("learning_record_target") or ""),
            regression_link=str(metadata.get("regression_link") or ""),
            durability_note=str(metadata.get("durability_note") or ""),
            review_contract_rows=list(metadata.get("patch_review_contract_rows") or []),
            closure_stage_rows=list(metadata.get("closure_stage_rows") or []),
            proof_queue=list(metadata.get("proof_queue") or []),
        )
        if recomputed != token:
            raise SystemExit(f"{label} learning closure token should be reproducible: {metadata}")
        tampered = _failure_learning_closure_token_sha256(
            target_test=str(metadata.get("target_test") or ""),
            target_file=str(metadata.get("target_file") or ""),
            patch_receipt_sha256=str(metadata.get("patch_receipt_sha256") or ""),
            learning_record_sha256=str(metadata.get("learning_record_sha256") or ""),
            apply_contract_sha256=str(metadata.get("apply_contract_sha256") or ""),
            failure_patch_review_token_sha256=str(metadata.get("failure_patch_review_token_sha256") or ""),
            failure_learning_record_token_sha256=str(metadata.get("failure_learning_record_token_sha256") or ""),
            after_action_learning=str(metadata.get("after_action_learning") or ""),
            learning_record_target=str(metadata.get("learning_record_target") or ""),
            regression_link=str(metadata.get("regression_link") or ""),
            durability_note=str(metadata.get("durability_note") or ""),
            review_contract_rows=list(metadata.get("patch_review_contract_rows") or []),
            closure_stage_rows=list(metadata.get("closure_stage_rows") or []),
            proof_queue=list(metadata.get("proof_queue") or []),
            authorizes_completion_claim=True,
        )
        if tampered == token:
            raise SystemExit(f"{label} learning closure token should bind completion-claim authority: {metadata}")
        tampered_record_token = _failure_learning_closure_token_sha256(
            target_test=str(metadata.get("target_test") or ""),
            target_file=str(metadata.get("target_file") or ""),
            patch_receipt_sha256=str(metadata.get("patch_receipt_sha256") or ""),
            learning_record_sha256=str(metadata.get("learning_record_sha256") or ""),
            apply_contract_sha256=str(metadata.get("apply_contract_sha256") or ""),
            failure_patch_review_token_sha256=str(metadata.get("failure_patch_review_token_sha256") or ""),
            failure_learning_record_token_sha256="0" * 64,
            after_action_learning=str(metadata.get("after_action_learning") or ""),
            learning_record_target=str(metadata.get("learning_record_target") or ""),
            regression_link=str(metadata.get("regression_link") or ""),
            durability_note=str(metadata.get("durability_note") or ""),
            review_contract_rows=list(metadata.get("patch_review_contract_rows") or []),
            closure_stage_rows=list(metadata.get("closure_stage_rows") or []),
            proof_queue=list(metadata.get("proof_queue") or []),
        )
        if tampered_record_token == token:
            raise SystemExit(f"{label} learning closure token should bind carried record token: {metadata}")
        tampered_review_token = _failure_learning_closure_token_sha256(
            target_test=str(metadata.get("target_test") or ""),
            target_file=str(metadata.get("target_file") or ""),
            patch_receipt_sha256=str(metadata.get("patch_receipt_sha256") or ""),
            learning_record_sha256=str(metadata.get("learning_record_sha256") or ""),
            apply_contract_sha256=str(metadata.get("apply_contract_sha256") or ""),
            failure_patch_review_token_sha256="0" * 64,
            failure_learning_record_token_sha256=str(metadata.get("failure_learning_record_token_sha256") or ""),
            after_action_learning=str(metadata.get("after_action_learning") or ""),
            learning_record_target=str(metadata.get("learning_record_target") or ""),
            regression_link=str(metadata.get("regression_link") or ""),
            durability_note=str(metadata.get("durability_note") or ""),
            review_contract_rows=list(metadata.get("patch_review_contract_rows") or []),
            closure_stage_rows=list(metadata.get("closure_stage_rows") or []),
            proof_queue=list(metadata.get("proof_queue") or []),
        )
        if tampered_review_token == token:
            raise SystemExit(f"{label} learning closure token should bind carried patch review token: {metadata}")
        tampered_stage_rows = [dict(row) for row in (metadata.get("closure_stage_rows") or [])]
        tampered_stage_rows[0]["authorizes_tool_execution"] = True
        tampered_stage_row = _failure_learning_closure_token_sha256(
            target_test=str(metadata.get("target_test") or ""),
            target_file=str(metadata.get("target_file") or ""),
            patch_receipt_sha256=str(metadata.get("patch_receipt_sha256") or ""),
            learning_record_sha256=str(metadata.get("learning_record_sha256") or ""),
            apply_contract_sha256=str(metadata.get("apply_contract_sha256") or ""),
            failure_patch_review_token_sha256=str(metadata.get("failure_patch_review_token_sha256") or ""),
            failure_learning_record_token_sha256=str(metadata.get("failure_learning_record_token_sha256") or ""),
            after_action_learning=str(metadata.get("after_action_learning") or ""),
            learning_record_target=str(metadata.get("learning_record_target") or ""),
            regression_link=str(metadata.get("regression_link") or ""),
            durability_note=str(metadata.get("durability_note") or ""),
            review_contract_rows=list(metadata.get("patch_review_contract_rows") or []),
            closure_stage_rows=tampered_stage_rows,
            proof_queue=list(metadata.get("proof_queue") or []),
        )
        if tampered_stage_row == token:
            raise SystemExit(f"{label} learning closure token should bind closure-stage authority flags: {metadata}")


def main() -> None:
    assert_feedback_metadata_bool_is_exact()
    assert_feedback_malformed_handoff_flags_are_exact()
    assert_planner_routes_feedback_phone_aliases()
    with TemporaryDirectory(prefix="jarvis-feedback-") as temp:
        runtime = make_temp_runtime(Path(temp))
        (
            record_feedback,
            feedback_report,
            save_feedback_report,
            feedback_actions,
            save_feedback_actions,
            failure_to_test_preview,
            repeated_failure_clusters,
            failure_promotion_packet,
            failure_implementation_packet,
            failure_apply_contract,
            failure_learning_cockpit,
            failure_patch_receipt_packet,
            failure_patch_application_bridge,
            failure_patch_completion_gate,
            failure_patch_handoff_packet,
            failure_patch_closeout_packet,
            failure_learning_record_packet,
            failure_learning_closure_ledger,
        ) = make_feedback_tools(
            runtime.store,
            runtime.vault,
        )
        patch_receipt_sha256 = "4" * 64
        learning_record_sha256 = "5" * 64
        apply_contract_sha256 = text_sha256(
            "\n".join(
                [
                    "failure_apply_contract_v1",
                    "frontend layout and visual regression",
                    "smoke_test_status_server_layout",
                    "jarvis_v2/scripts/smoke_test_status_server.py",
                    "python3 -m jarvis_v2.scripts.smoke_test_status_server",
                ]
            )
        )
        ready_patch_summary = (
            "layout; changed files jarvis_v2/scripts/smoke_test_status_server.py; "
            "test first pre-patch failing assertion reviewed; "
            "verification python3 -m jarvis_v2.scripts.smoke_test_status_server passed; "
            "compile python3 -m compileall -q jarvis_v2 passed; "
            "rollback scoped assertion can be reverted; "
            f"patch receipt sha256 {patch_receipt_sha256}; "
            f"apply contract sha256 {apply_contract_sha256}; "
            "contract apply contract reviewed; "
            "application bridge applied patch bound to contract"
        )
        ready_learning_summary = (
            ready_patch_summary
            + "; completion audit reviewed; "
            "evidence ledger reviewed; "
            "completion claim gate reviewed; "
            "review post-claim review complete; "
            "after-action learning reviewed; "
            "record target learning review; "
            "regression smoke_test_status_server_layout linked; "
            "durability checkpoint saved; "
            f"learning record sha256 {learning_record_sha256}"
        )
        cases = [
            "feedback: Jarvis should explain approvals more clearly before risky actions",
            "Jarvis feedback: conversational answers should be shorter when I ask what changed",
            "feedback: Jarvis dashboard text overlapped and hid diagnostics",
            "feedback: Jarvis dashboard layout overlapped the composer",
            "feedback report",
            "save feedback report",
            "feedback actions",
            "save feedback actions",
            "failure to test: Jarvis overlapped dashboard text and hid diagnostics",
            "failure clusters",
            "failure promotion packet",
            "failure implementation packet: layout",
            "failure apply contract: layout",
            f"failure patch receipt: {ready_patch_summary}",
            f"failure patch application bridge: {ready_patch_summary}",
            f"failure patch completion gate: {ready_patch_summary}",
            f"failure patch handoff: {ready_patch_summary}",
            f"failure patch closeout: {ready_learning_summary}",
            f"failure learning record: {ready_learning_summary}",
            f"failure learning closure ledger: {ready_learning_summary}",
            "search memory for approvals more clearly",
            "help memory",
            "list tools feedback",
            "list tools learning",
        ]
        report_note = Path(temp) / "Vault" / "Jarvis" / "Automations" / "Feedback Report.md"
        actions_note = Path(temp) / "Vault" / "Jarvis" / "Automations" / "Feedback Actions.md"
        for case in cases:
            result = runtime.handle(case)
            print(f"[{'ok' if result.verified else 'blocked'}] {case}")
            print(result.response[:1800])
            print()
            if not result.verified:
                raise SystemExit(f"Expected '{case}' to run.")
            assert_feedback_result_no_future_authority(result, case)
            if case == "feedback report":
                assert_contains(
                    result.response,
                    [
                        "Jarvis feedback report",
                        "Recent feedback",
                        "approvals more clearly",
                        "conversational answers should be shorter",
                        "Themes",
                        "safety",
                        "conversation",
                        "Safe improvement paths",
                        "set preference",
                        "approval-gated",
                    ],
                    case,
                )
            if case == "save feedback report":
                assert_vault_relative_receipt(result, Path(temp), "Automations/", case)
                assert_contains(
                    result.response,
                    [
                        "Feedback report saved",
                        "Jarvis feedback report",
                        "Recent feedback",
                        "approvals more clearly",
                        "Themes",
                        "Safe improvement paths",
                    ],
                    case,
                )
                if not report_note.exists():
                    raise SystemExit(f"{case} did not write expected note: {report_note}")
                assert_contains(
                    report_note.read_text(encoding="utf-8"),
                    [
                        "# Feedback Report",
                        "Jarvis feedback report",
                        "Recent feedback",
                        "conversation",
                        "safety",
                    ],
                    f"{case} note",
                )
            if case == "feedback actions":
                assert_contains(
                    result.response,
                    [
                        "Jarvis feedback actions",
                        "reviewable suggestions only",
                        "Suggested preference updates",
                        "set preference response length",
                        "set preference safety explanations",
                        "Skill or workflow candidates",
                        "approval-explanation skill",
                        "Test/code candidates",
                        "safety smoke test",
                        "Still approval-gated",
                    ],
                    case,
                )
            if case.startswith("failure patch receipt"):
                assert_contains(
                    result.response,
                    [
                        "Jarvis failure patch receipt packet",
                        "PATCH_RECEIPT_READY_FOR_COMPLETION_REVIEW",
                        "Receipt checklist",
                        "focused verification passed: yes",
                        "compile pass present: yes",
                        "rollback note present: yes",
                    ],
                    case,
                )
                metadata = result.tool_results[0].metadata
                if metadata.get("tool_name") == "failure_patch_receipt_packet":
                    raise SystemExit("ToolResult metadata should not shadow tool name.")
                if metadata.get("patch_receipt_ready") is not True or metadata.get("completion_review_ready") is not True:
                    raise SystemExit(f"failure patch receipt route missed ready metadata: {metadata}")
                assert_exact_test_contract(metadata, "failure patch receipt route")
            if case.startswith("failure patch application bridge"):
                assert_contains(
                    result.response,
                    [
                        "Jarvis failure patch application bridge",
                        "FAILURE_PATCH_APPLICATION_BRIDGE_READY",
                        "Bridge state",
                        "ready for completion gate: yes",
                        "apply contract reviewed: yes",
                        "applied patch bound to contract: yes",
                        "patch receipt ready: yes",
                        "focused verification present: yes",
                        "compile pass present: yes",
                        "rollback note present: yes",
                    ],
                    case,
                )
                metadata = result.tool_results[0].metadata
                if metadata.get("application_bridge_ready") is not True or metadata.get("applied_patch_bound_to_contract") is not True:
                    raise SystemExit(f"failure patch application bridge route missed ready metadata: {metadata}")
                assert_exact_test_contract(metadata, "failure patch application bridge route")
            if case.startswith("failure patch completion gate"):
                assert_contains(
                    result.response,
                    [
                        "Jarvis failure patch completion gate",
                        "FAILURE_PATCH_COMPLETION_READY_FOR_CLAIM_REVIEW",
                        "Evidence checks",
                        "apply contract reviewed: yes",
                        "application bridge ready: yes",
                        "applied patch bound to contract: yes",
                        "patch receipt ready: yes",
                        "focused verification present: yes",
                        "compile pass present: yes",
                        "rollback note present: yes",
                        "completion claim gate",
                    ],
                    case,
                )
                metadata = result.tool_results[0].metadata
                if metadata.get("completion_claim_ready") is not True or metadata.get("ready_for_completion_claim_review") is not True:
                    raise SystemExit(f"failure patch completion gate route missed ready metadata: {metadata}")
                if metadata.get("application_bridge_ready") is not True or metadata.get("applied_patch_bound_to_contract") is not True:
                    raise SystemExit(f"failure patch completion gate missed bridge metadata: {metadata}")
                assert_exact_test_contract(metadata, "failure patch completion gate route")
            if case.startswith("failure patch handoff"):
                assert_contains(
                    result.response,
                    [
                        "Jarvis failure patch handoff packet",
                        "FAILURE_PATCH_HANDOFF_READY_FOR_COMPLETION_REVIEW",
                        "Handoff state",
                        "ready for completion review: yes",
                        "completion gate state",
                        "apply contract reviewed: yes",
                        "application bridge ready: yes",
                        "applied patch bound to contract: yes",
                        "patch receipt ready: yes",
                        "focused verification present: yes",
                        "compile pass present: yes",
                        "rollback note present: yes",
                        "Pre-claim proof chain",
                        "Post-claim review queue",
                        "completion audit",
                        "evidence ledger",
                        "completion claim gate",
                    ],
                    case,
                )
                metadata = result.tool_results[0].metadata
                if metadata.get("ready_for_completion_review") is not True or metadata.get("handoff_state") != "FAILURE_PATCH_HANDOFF_READY_FOR_COMPLETION_REVIEW":
                    raise SystemExit(f"failure patch handoff route missed ready metadata: {metadata}")
                assert_exact_test_contract(metadata, "failure patch handoff route")
            if case.startswith("failure patch closeout"):
                assert_contains(
                    result.response,
                    [
                        "Jarvis failure patch closeout packet",
                        "FAILURE_PATCH_CLOSEOUT_READY_FOR_LEARNING_RECORD",
                        "Closeout state",
                        "ready for learning record: yes",
                        "completion audit reviewed: yes",
                        "evidence ledger reviewed: yes",
                        "completion claim gate reviewed: yes",
                        "post-claim review complete: yes",
                        "Learning record queue",
                    ],
                    case,
                )
                metadata = result.tool_results[0].metadata
                if metadata.get("ready_for_learning_record") is not True or metadata.get("closeout_state") != "FAILURE_PATCH_CLOSEOUT_READY_FOR_LEARNING_RECORD":
                    raise SystemExit(f"failure patch closeout route missed ready metadata: {metadata}")
                assert_exact_test_contract(metadata, "failure patch closeout route")
            if case.startswith("failure learning record"):
                assert_contains(
                    result.response,
                    [
                        "Jarvis failure learning record packet",
                        "FAILURE_LEARNING_RECORD_READY",
                        "Learning record state",
                        "ready as durable learning record: yes",
                        "after-action learning reviewed: yes",
                        "learning record target: learning review",
                        "regression linked: yes",
                        "durable record evidence: yes",
                        "failure learning record token sha256",
                        "record token authorizes completion claim: no",
                    ],
                    case,
                )
                metadata = result.tool_results[0].metadata
                if metadata.get("ready_for_durable_learning_record") is not True or metadata.get("record_state") != "FAILURE_LEARNING_RECORD_READY":
                    raise SystemExit(f"failure learning record route missed ready metadata: {metadata}")
                assert_exact_test_contract(metadata, "failure learning record route")
                assert_failure_learning_record_token(metadata, "failure learning record route")
            if case.startswith("failure learning closure ledger"):
                assert_contains(
                    result.response,
                    [
                        "Jarvis failure learning closure ledger",
                        "FAILURE_LEARNING_CLOSURE_READY",
                        "ready as durable learning closure: yes",
                        "failure learning record token sha256",
                        "record token authorizes completion claim: no",
                        "Closure stages",
                        "learning_record: ready",
                        "Proof queue",
                        "Boundary",
                    ],
                    case,
                )
                metadata = result.tool_results[0].metadata
                if metadata.get("ready_as_durable_learning_closure") is not True or metadata.get("ledger_state") != "FAILURE_LEARNING_CLOSURE_READY":
                    raise SystemExit(f"failure learning closure ledger route missed ready metadata: {metadata}")
                assert_exact_test_contract(metadata, "failure learning closure ledger route")
                assert_failure_learning_record_token(metadata, "failure learning closure ledger route")
            if case == "save feedback actions":
                assert_vault_relative_receipt(result, Path(temp), "Automations/", case)
                assert_contains(
                    result.response,
                    [
                        "Feedback actions saved",
                        "Jarvis feedback actions",
                        "Suggested preference updates",
                        "Still approval-gated",
                    ],
                    case,
                )
                if not actions_note.exists():
                    raise SystemExit(f"{case} did not write expected note: {actions_note}")
                assert_contains(
                    actions_note.read_text(encoding="utf-8"),
                    [
                        "# Feedback Actions",
                        "Jarvis feedback actions",
                        "set preference response length",
                        "set preference safety explanations",
                        "approval-gated",
                    ],
                    f"{case} note",
                )
            if case.startswith("failure to test:"):
                assert_contains(
                    result.response,
                    [
                        "Jarvis failure-to-test preview",
                        "Proposed smoke test",
                        "smoke_test_status_server_layout",
                        "Proposed assertions",
                        "Safe next commands",
                        "Boundary",
                        "Read-only preview only",
                    ],
                    case,
                )
            if case == "failure clusters":
                assert_contains(
                    result.response,
                    [
                        "Jarvis repeated-failure cluster report",
                        "read-only",
                        "frontend layout and visual regression",
                        "strong repeated signal",
                        "smoke_test_status_server_layout",
                        "target file",
                        "expected behavior",
                        "observed failure",
                        "rollback",
                        "Ranked next safe move",
                        "Boundary",
                    ],
                    case,
                )
            if case == "failure promotion packet":
                assert_contains(
                    result.response,
                    [
                        "Jarvis failure promotion packet",
                        "read-only promotion plan",
                        "Selected cluster",
                        "target smoke test",
                        "smoke_test_status_server_layout",
                        "likely file",
                        "Draft smoke-test assertions",
                        "Promotion checklist",
                        "Boundary",
                    ],
                    case,
                )
            if case == "failure implementation packet: layout":
                assert_contains(
                    result.response,
                    [
                        "Jarvis failure implementation packet",
                        "implementation-ready",
                        "Selected failure cluster",
                        "target file",
                        "jarvis_v2/scripts/smoke_test_status_server.py",
                        "focused verification",
                        "python3 -m jarvis_v2.scripts.smoke_test_status_server",
                        "Required assertions",
                        "Implementation steps",
                        "Stop conditions",
                        "Verification sequence",
                        "Boundary",
                    ],
                    case,
                )
            if case == "failure apply contract: layout":
                assert_contains(
                    result.response,
                    [
                        "Jarvis failure apply contract",
                        "last-look contract",
                        "Selected cluster",
                        "target file",
                        "jarvis_v2/scripts/smoke_test_status_server.py",
                        "focused verification",
                        "python3 -m jarvis_v2.scripts.smoke_test_status_server",
                        "Required assertions",
                        "Apply sequence",
                        "Approval and safety rules",
                        "Rollback plan",
                        "Completion receipt",
                        "Boundary",
                    ],
                    case,
                )
            if case == "search memory for approvals more clearly":
                assert_contains(result.response, ["feedback", "approvals more clearly"], case)
            if case == "help memory":
                assert_contains(result.response, ["feedback:", "failure to test:", "failure clusters", "failure promotion packet", "failure implementation packet", "failure apply contract", "failure patch receipt", "failure patch application bridge", "failure patch completion gate", "failure patch handoff", "failure patch closeout", "failure learning record", "failure learning closure ledger", "feedback report", "save feedback report", "feedback actions", "save feedback actions"], case)
            if case == "list tools feedback":
                assert_contains(result.response, ["record_feedback", "feedback_report", "save_feedback_report", "feedback_actions", "save_feedback_actions"], case)
            if case == "list tools learning":
                assert_contains(result.response, ["failure_to_test_preview", "repeated_failure_clusters", "failure_promotion_packet", "failure_implementation_packet", "failure_apply_contract", "failure_patch_receipt_packet", "failure_patch_application_bridge", "failure_patch_completion_gate", "failure_patch_handoff_packet", "failure_patch_closeout_packet", "failure_learning_record_packet", "failure_learning_closure_ledger"], case)

        empty_body = record_feedback({"body": ""})
        if empty_body.ok or empty_body.metadata.get("reason") != "missing_body":
            raise SystemExit("record_feedback should reject empty feedback with safe metadata.")

        oversized_body = record_feedback({"body": "x" * 50001})
        if oversized_body.ok or "too large" not in oversized_body.output:
            raise SystemExit("record_feedback should refuse oversized feedback.")
        if oversized_body.metadata.get("reason") != "body_too_large" or oversized_body.metadata.get("writes_files"):
            raise SystemExit("record_feedback oversized refusal missed safe metadata.")
        if oversized_body.metadata.get("memory_id") is not None:
            raise SystemExit("record_feedback oversized refusal should expose stable memory_id=None metadata.")

        record_result = record_feedback({"body": "Jarvis should keep feedback safe and reviewable."})
        if not record_result.metadata.get("writes_files") or not record_result.metadata.get("writes_database"):
            raise SystemExit("record_feedback should mark local file and database writes.")
        if not record_result.metadata.get("writes_memory") or not record_result.metadata.get("writes_notes"):
            raise SystemExit("record_feedback should mark memory and note writes.")
        if record_result.metadata.get("queues_approval") or record_result.metadata.get("controls_computer"):
            raise SystemExit("record_feedback should not queue approvals or control the computer.")
        assert_basic_feedback_handoff(
            record_result.metadata,
            "feedback_capture_handoff",
            "record_feedback direct",
            state_changed=True,
            changed=["feedback"],
            content_in_handoff=False,
            writes_files=True,
            writes_memory=True,
            writes_notes=True,
            writes_database=True,
        )

        before_path_feedback = len(runtime.store.recent_memories(limit=1000))
        path_body = record_feedback({"body": "Jarvis leaked /\x55sers/example/Desktop/Claude code/private-feedback.md into a report."})
        if path_body.ok or path_body.metadata.get("reason") != "invalid_body" or path_body.metadata.get("memory_id") is not None:
            raise SystemExit(f"record_feedback should reject path-shaped feedback bodies: {path_body.metadata}")
        if path_body.metadata.get("writes_files") or path_body.metadata.get("writes_database"):
            raise SystemExit("path-shaped feedback body should not mark durable writes.")
        if "<local-path>" not in str(path_body.metadata.get("raw_body")):
            raise SystemExit(f"path-shaped feedback body should be redacted in metadata: {path_body.metadata}")
        if "/\x55sers/" in path_body.output or "/\x55sers/" in str(path_body.metadata):
            raise SystemExit("path-shaped feedback body should not echo raw local paths.")
        for value in [
            "Jarvis leaked /var/folders/zc/feedback-body.md into a report.",
            "Jarvis leaked /tmp/feedback-body.md into a report.",
        ]:
            path_temp_body = record_feedback({"body": value})
            if path_temp_body.ok or path_temp_body.metadata.get("reason") != "invalid_body" or path_temp_body.metadata.get("memory_id") is not None:
                raise SystemExit(f"record_feedback should reject temp-root feedback bodies: {path_temp_body.metadata}")
            if path_temp_body.metadata.get("writes_files") or path_temp_body.metadata.get("writes_database"):
                raise SystemExit("temp-root feedback body should not mark durable writes.")
            if "<local-path>" not in str(path_temp_body.metadata.get("raw_body")):
                raise SystemExit(f"temp-root feedback body should be redacted in metadata: {path_temp_body.metadata}")
            assert_no_local_path(path_temp_body.output + str(path_temp_body.metadata), "temp-root feedback body refusal")

        path_title = record_feedback({"title": "/private/tmp/jarvis-feedback-title.md", "body": "Jarvis feedback title should be safe."})
        if path_title.ok or path_title.metadata.get("reason") != "invalid_title" or path_title.metadata.get("memory_id") is not None:
            raise SystemExit(f"record_feedback should reject path-shaped feedback titles: {path_title.metadata}")
        if path_title.metadata.get("writes_files") or path_title.metadata.get("writes_database"):
            raise SystemExit("path-shaped feedback title should not mark durable writes.")
        if "<local-path>" not in str(path_title.metadata.get("raw_title")):
            raise SystemExit(f"path-shaped feedback title should be redacted in metadata: {path_title.metadata}")
        if "/private/" in path_title.output or "/private/" in str(path_title.metadata):
            raise SystemExit("path-shaped feedback title should not echo raw local paths.")
        for value in ["/var/folders/zc/jarvis-feedback-title.md", "/tmp/jarvis-feedback-title.md"]:
            path_temp_title = record_feedback({"title": value, "body": "Jarvis feedback title should be safe."})
            if path_temp_title.ok or path_temp_title.metadata.get("reason") != "invalid_title" or path_temp_title.metadata.get("memory_id") is not None:
                raise SystemExit(f"record_feedback should reject temp-root feedback titles: {path_temp_title.metadata}")
            if path_temp_title.metadata.get("writes_files") or path_temp_title.metadata.get("writes_database"):
                raise SystemExit("temp-root feedback title should not mark durable writes.")
            if path_temp_title.metadata.get("raw_title") != "<local-path>":
                raise SystemExit(f"temp-root feedback title should be redacted in metadata: {path_temp_title.metadata}")
            assert_no_local_path(path_temp_title.output + str(path_temp_title.metadata), "temp-root feedback title refusal")

        path_theme = record_feedback({"theme": "/\x55sers/example/Desktop/theme", "body": "Jarvis feedback theme should be safe."})
        if path_theme.ok or path_theme.metadata.get("reason") != "invalid_theme" or path_theme.metadata.get("memory_id") is not None:
            raise SystemExit(f"record_feedback should reject path-shaped feedback themes: {path_theme.metadata}")
        if path_theme.metadata.get("writes_files") or path_theme.metadata.get("writes_database"):
            raise SystemExit("path-shaped feedback theme should not mark durable writes.")
        if "<local-path>" not in str(path_theme.metadata.get("raw_theme")):
            raise SystemExit(f"path-shaped feedback theme should be redacted in metadata: {path_theme.metadata}")
        if "/\x55sers/" in path_theme.output or "/\x55sers/" in str(path_theme.metadata):
            raise SystemExit("path-shaped feedback theme should not echo raw local paths.")
        for value in ["/var/folders/zc/feedback-theme", "/tmp/feedback-theme"]:
            path_temp_theme = record_feedback({"theme": value, "body": "Jarvis feedback theme should be safe."})
            if path_temp_theme.ok or path_temp_theme.metadata.get("reason") != "invalid_theme" or path_temp_theme.metadata.get("memory_id") is not None:
                raise SystemExit(f"record_feedback should reject temp-root feedback themes: {path_temp_theme.metadata}")
            if path_temp_theme.metadata.get("writes_files") or path_temp_theme.metadata.get("writes_database"):
                raise SystemExit("temp-root feedback theme should not mark durable writes.")
            if path_temp_theme.metadata.get("raw_theme") != "<local-path>":
                raise SystemExit(f"temp-root feedback theme should be redacted in metadata: {path_temp_theme.metadata}")
            assert_no_local_path(path_temp_theme.output + str(path_temp_theme.metadata), "temp-root feedback theme refusal")
        after_path_feedback = len(runtime.store.recent_memories(limit=1000))
        if before_path_feedback != after_path_feedback:
            raise SystemExit("path-shaped feedback inputs should not create memories.")

        report_result = feedback_report({"limit": "not-a-number"})
        if report_result.metadata.get("limit") != 12 or report_result.metadata.get("writes_files"):
            raise SystemExit("feedback_report should sanitize bad limits and remain read-only.")
        assert_basic_feedback_handoff(
            report_result.metadata,
            "feedback_report_handoff",
            "feedback_report direct",
            state_changed=False,
            changed=[],
            content_in_handoff=True,
            writes_files=False,
            writes_memory=False,
            writes_notes=False,
            writes_database=False,
        )

        bool_report = feedback_report({"limit": True})
        if bool_report.metadata.get("limit") != 12 or bool_report.metadata.get("writes_files"):
            raise SystemExit(f"feedback_report should treat boolean limits as malformed defaults: {bool_report.metadata}")

        clipped_report = feedback_report({"limit": 999999})
        if clipped_report.metadata.get("limit") != 200:
            raise SystemExit("feedback_report should clamp huge limits.")

        low_actions = feedback_actions({"limit": -5})
        if low_actions.metadata.get("limit") != 1 or low_actions.metadata.get("writes_files"):
            raise SystemExit("feedback_actions should clamp low limits and remain read-only.")
        assert_basic_feedback_handoff(
            low_actions.metadata,
            "feedback_actions_handoff",
            "feedback_actions direct",
            state_changed=False,
            changed=[],
            content_in_handoff=True,
            writes_files=False,
            writes_memory=False,
            writes_notes=False,
            writes_database=False,
        )

        bool_actions = feedback_actions({"limit": False})
        if bool_actions.metadata.get("limit") != 12 or bool_actions.metadata.get("writes_files"):
            raise SystemExit(f"feedback_actions should treat boolean limits as malformed defaults: {bool_actions.metadata}")

        saved_report = save_feedback_report({"limit": "bad"})
        if saved_report.metadata.get("limit") != 12 or not saved_report.metadata.get("writes_files"):
            raise SystemExit("save_feedback_report should sanitize limits and mark file writes.")
        if not saved_report.metadata.get("writes_notes") or saved_report.metadata.get("writes_memory"):
            raise SystemExit("save_feedback_report should mark note writes without memory writes.")
        assert_vault_relative_receipt(saved_report, Path(temp), "Automations/", "save_feedback_report direct")
        assert_no_local_path(saved_report.output, "save_feedback_report output")
        assert_basic_feedback_handoff(
            saved_report.metadata,
            "feedback_report_handoff",
            "save_feedback_report direct",
            state_changed=True,
            changed=["feedback_report"],
            content_in_handoff=True,
            writes_files=True,
            writes_memory=False,
            writes_notes=True,
            writes_database=False,
        )

        saved_actions = save_feedback_actions({"limit": 50000})
        if saved_actions.metadata.get("limit") != 200 or not saved_actions.metadata.get("writes_files"):
            raise SystemExit("save_feedback_actions should clamp limits and mark file writes.")
        if not saved_actions.metadata.get("writes_notes") or saved_actions.metadata.get("writes_memory"):
            raise SystemExit("save_feedback_actions should mark note writes without memory writes.")
        assert_vault_relative_receipt(saved_actions, Path(temp), "Automations/", "save_feedback_actions direct")
        assert_basic_feedback_handoff(
            saved_actions.metadata,
            "feedback_actions_handoff",
            "save_feedback_actions direct",
            state_changed=True,
            changed=["feedback_actions"],
            content_in_handoff=True,
            writes_files=True,
            writes_memory=False,
            writes_notes=True,
            writes_database=False,
        )

        bounded_title = record_feedback({"title": "t" * 500, "theme": "conversation", "body": "Jarvis feedback titles are bounded."})
        if not bounded_title.ok or bounded_title.metadata.get("title_chars") != 160:
            raise SystemExit("record_feedback should bound long titles.")

        preview_result = failure_to_test_preview({"failure": "Jarvis hid diagnostics under the command box."})
        if not preview_result.ok or preview_result.metadata.get("suggested_test") != "smoke_test_status_server_layout":
            raise SystemExit("failure_to_test_preview should identify layout failures.")
        if preview_result.metadata.get("writes_files") or preview_result.metadata.get("queues_approval"):
            raise SystemExit("failure_to_test_preview should remain read-only.")

        path_preview = failure_to_test_preview(
            {
                "failure": "Jarvis hid diagnostics under /\x55sers/example/private/dashboard.png and /private/tmp/screenshot.png."
            }
        )
        if not path_preview.ok or path_preview.metadata.get("suggested_test") != "smoke_test_status_server_layout":
            raise SystemExit("failure_to_test_preview should keep classifying redacted path-seeded layout failures.")
        if "<local-path>" not in path_preview.output:
            raise SystemExit(f"failure_to_test_preview should redact path-shaped failure summaries: {path_preview.output}")
        assert_no_local_path(path_preview.output + str(path_preview.metadata), "failure_to_test_preview path-seeded output")
        if path_preview.metadata.get("writes_files") or path_preview.metadata.get("queues_approval"):
            raise SystemExit("path-seeded failure_to_test_preview should remain read-only.")

        empty_preview = failure_to_test_preview({"failure": ""})
        if empty_preview.ok or empty_preview.metadata.get("reason") != "missing_failure":
            raise SystemExit("failure_to_test_preview should reject empty failures safely.")

        cluster_result = repeated_failure_clusters({"limit": 999999})
        if not cluster_result.ok or cluster_result.metadata.get("limit") != 200:
            raise SystemExit("repeated_failure_clusters should clamp limits and run.")
        if cluster_result.metadata.get("repeated_clusters", 0) < 1 or cluster_result.metadata.get("top_test") != "smoke_test_status_server_layout":
            raise SystemExit("repeated_failure_clusters should find the repeated layout cluster.")
        if cluster_result.metadata.get("top_target_file") != "jarvis_v2/scripts/smoke_test_status_server.py":
            raise SystemExit(f"repeated_failure_clusters missed target file metadata: {cluster_result.metadata}")
        if not cluster_result.metadata.get("top_observed_failure") or "dashboard" not in cluster_result.metadata.get("top_observed_failure", ""):
            raise SystemExit(f"repeated_failure_clusters missed observed failure metadata: {cluster_result.metadata}")

        bool_cluster = repeated_failure_clusters({"limit": True})
        if not bool_cluster.ok or bool_cluster.metadata.get("limit") != 24:
            raise SystemExit(f"repeated_failure_clusters should treat boolean limits as malformed defaults: {bool_cluster.metadata}")
        if not cluster_result.metadata.get("top_expected_behavior") or "dashboard" not in cluster_result.metadata.get("top_expected_behavior", "").lower():
            raise SystemExit(f"repeated_failure_clusters missed expected behavior metadata: {cluster_result.metadata}")
        if not cluster_result.metadata.get("top_rollback_note") or "Revert" not in cluster_result.metadata.get("top_rollback_note", ""):
            raise SystemExit(f"repeated_failure_clusters missed rollback metadata: {cluster_result.metadata}")
        cluster_rows = cluster_result.metadata.get("cluster_rows") or []
        if not cluster_rows or not cluster_rows[0].get("observed_failures") or not cluster_rows[0].get("expected_behavior") or not cluster_rows[0].get("rollback_note"):
            raise SystemExit(f"repeated_failure_clusters missed structured exact-test handoff rows: {cluster_result.metadata}")
        assert_exact_test_contract(cluster_rows[0], "repeated_failure_clusters top row")
        for row in cluster_rows:
            target_file = row.get("target_file")
            if not target_file or not Path(target_file).exists():
                raise SystemExit(f"repeated_failure_clusters produced a missing target file: {row}")
        if cluster_result.metadata.get("writes_files") or cluster_result.metadata.get("queues_approval"):
            raise SystemExit("repeated_failure_clusters should remain read-only.")

        runtime.store.add_memory(
            MemoryRecord(
                "feedback",
                "/var/folders/zc/legacy-feedback-title",
                "Legacy feedback body leaked /\x55sers/example/private/feedback.md and /tmp/feedback.md.",
                "smoke",
            )
        )
        legacy_report = feedback_report({"limit": 12})
        if "<local-path>" not in legacy_report.output:
            raise SystemExit(f"feedback_report should scrub legacy path-bearing feedback rows: {legacy_report.output}")
        assert_no_local_path(legacy_report.output, "feedback_report legacy output")
        legacy_saved_report = save_feedback_report({"limit": 12})
        assert_no_local_path(legacy_saved_report.output, "legacy save_feedback_report output")
        legacy_saved_report_body = Path(legacy_saved_report.metadata["path"]).read_text(encoding="utf-8")
        if "<local-path>" not in legacy_saved_report_body:
            raise SystemExit(f"saved feedback report should preserve redacted markers: {legacy_saved_report_body}")
        assert_no_local_path(legacy_saved_report_body, "legacy saved feedback report body")

        malformed_runtime = make_temp_runtime(Path(temp) / "malformed-feedback")
        (
            _malformed_record_feedback,
            malformed_feedback_report,
            malformed_save_feedback_report,
            malformed_feedback_actions,
            malformed_save_feedback_actions,
            _malformed_failure_to_test_preview,
            malformed_repeated_failure_clusters,
            malformed_failure_promotion_packet,
            *_malformed_rest,
        ) = make_feedback_tools(malformed_runtime.store, malformed_runtime.vault)
        hostile_marker = "FEEDBACK_HOSTILE_ROW_SHOULD_NOT_LEAK /\x55sers/example/private/feedback.sqlite"
        feedback_rows = [
            HostileRow(hostile_marker),
            {
                "id": 91,
                "category": "feedback",
                "title": "/private/tmp/feedback-title",
                "body": "Jarvis dashboard text overlapped diagnostics /\x55sers/example/private/feedback-body",
            },
            {
                "id": 92,
                "category": "feedback",
                "title": "Dashboard layout miss",
                "body": "Jarvis dashboard layout overlapped the composer panel /var/folders/zc/feedback-body",
            },
        ]
        malformed_runtime.store.recent_memories = lambda limit=100: feedback_rows[:limit]
        malformed_runtime.store.list_memories = lambda category=None, limit=100: feedback_rows[:limit]
        malformed_report = malformed_feedback_report({"limit": 10})
        malformed_actions = malformed_feedback_actions({"limit": 10})
        malformed_clusters = malformed_repeated_failure_clusters({"limit": 10})
        malformed_promotion = malformed_failure_promotion_packet({"cluster": "layout", "limit": 10})
        for label, result in (
            ("malformed feedback_report", malformed_report),
            ("malformed feedback_actions", malformed_actions),
            ("malformed repeated_failure_clusters", malformed_clusters),
            ("malformed failure_promotion_packet", malformed_promotion),
        ):
            if not result.ok:
                raise SystemExit(f"{label} should tolerate malformed local feedback rows: {result.output}")
            combined = result.output + repr(result.metadata)
            if "FEEDBACK_HOSTILE_ROW_SHOULD_NOT_LEAK" in combined or "feedback.sqlite" in combined:
                raise SystemExit(f"{label} leaked hostile row marker: {combined}")
            assert_no_local_path(combined, label)
            if result.metadata.get("readable_feedback_rows") != 2 or result.metadata.get("unreadable_feedback_rows") != 1:
                raise SystemExit(f"{label} missed readable/unreadable feedback counts: {result.metadata}")
        for label, result, handoff_key in (
            ("malformed feedback_report", malformed_report, "feedback_report_handoff"),
            ("malformed feedback_actions", malformed_actions, "feedback_actions_handoff"),
        ):
            handoff = result.metadata.get(handoff_key) or {}
            if handoff.get("readable_feedback_rows") != 2 or handoff.get("unreadable_feedback_rows") != 1:
                raise SystemExit(f"{label} handoff missed readable/unreadable counts: {handoff}")
            if "hidden malformed feedback rows: 1" not in result.output:
                raise SystemExit(f"{label} missed safe hidden-row diagnostic: {result.output}")
            assert_basic_feedback_handoff(
                result.metadata,
                handoff_key,
                label,
                state_changed=False,
                changed=[],
                content_in_handoff=True,
                writes_files=False,
                writes_memory=False,
                writes_notes=False,
                writes_database=False,
            )
        for label, result, handoff_key, changed in (
            (
                "malformed save_feedback_report",
                malformed_save_feedback_report({"limit": 10}),
                "feedback_report_handoff",
                ["feedback_report"],
            ),
            (
                "malformed save_feedback_actions",
                malformed_save_feedback_actions({"limit": 10}),
                "feedback_actions_handoff",
                ["feedback_actions"],
            ),
        ):
            if result.metadata.get("readable_feedback_rows") != 2 or result.metadata.get("unreadable_feedback_rows") != 1:
                raise SystemExit(f"{label} lost readable/unreadable feedback counts: {result.metadata}")
            handoff = result.metadata.get(handoff_key) or {}
            if handoff.get("unreadable_feedback_rows") != 1:
                raise SystemExit(f"{label} handoff lost malformed-row evidence: {handoff}")
            assert_basic_feedback_handoff(
                result.metadata,
                handoff_key,
                label,
                state_changed=True,
                changed=changed,
                content_in_handoff=True,
                writes_files=True,
                writes_memory=False,
                writes_notes=True,
                writes_database=False,
            )
        if "hidden malformed feedback rows: 1" not in malformed_clusters.output:
            raise SystemExit(f"malformed repeated_failure_clusters missed hidden-row diagnostic: {malformed_clusters.output}")

        bounded_runtime = make_temp_runtime(Path(temp) / "bounded-feedback")
        for index in range(150):
            bounded_runtime.store.add_memory(
                MemoryRecord("feedback", f"Feedback {index}", f"Conversation feedback {index}", "smoke")
            )
        for index in range(120):
            bounded_runtime.store.add_memory(
                MemoryRecord("facts", f"Newer fact {index}", f"Non-feedback memory {index}", "smoke")
            )
        _, bounded_feedback_report, *_ = make_feedback_tools(bounded_runtime.store, bounded_runtime.vault)
        bounded_result = bounded_feedback_report({"limit": 200})
        if bounded_result.metadata.get("count") != 150:
            raise SystemExit(
                "feedback_report should honor its 200-row limit after category filtering: "
                f"{bounded_result.metadata}"
            )
        if malformed_clusters.metadata.get("repeated_clusters", 0) < 1 or malformed_clusters.metadata.get("top_test") != "smoke_test_status_server_layout":
            raise SystemExit(f"malformed repeated_failure_clusters missed readable repeated signal: {malformed_clusters.metadata}")
        if malformed_promotion.metadata.get("target_test") != "smoke_test_status_server_layout":
            raise SystemExit(f"malformed failure_promotion_packet missed readable target test: {malformed_promotion.metadata}")

        promotion_result = failure_promotion_packet({"cluster": "layout"})
        if not promotion_result.ok or promotion_result.metadata.get("target_test") != "smoke_test_status_server_layout":
            raise SystemExit("failure_promotion_packet should select the repeated layout cluster.")
        if promotion_result.metadata.get("promoted") or promotion_result.metadata.get("writes_files") or promotion_result.metadata.get("queues_approval"):
            raise SystemExit("failure_promotion_packet should stay read-only and not promote automatically.")
        if not promotion_result.metadata.get("observed_failures") or not promotion_result.metadata.get("expected_behavior") or not promotion_result.metadata.get("rollback_note"):
            raise SystemExit(f"failure_promotion_packet missed exact-test handoff metadata: {promotion_result.metadata}")
        assert_exact_test_contract(promotion_result.metadata, "failure_promotion_packet")

        implementation_result = failure_implementation_packet({"cluster": "layout"})
        if not implementation_result.ok or implementation_result.metadata.get("target_test") != "smoke_test_status_server_layout":
            raise SystemExit("failure_implementation_packet should select the repeated layout cluster.")
        if implementation_result.metadata.get("target_file") != "jarvis_v2/scripts/smoke_test_status_server.py":
            raise SystemExit("failure_implementation_packet should choose the dashboard smoke test file.")
        if not implementation_result.metadata.get("implementation_ready"):
            raise SystemExit("failure_implementation_packet should mark repeated clusters implementation-ready.")
        if implementation_result.metadata.get("writes_files") or implementation_result.metadata.get("queues_approval"):
            raise SystemExit("failure_implementation_packet should stay read-only.")
        assert_exact_test_contract(implementation_result.metadata, "failure_implementation_packet")

        apply_result = failure_apply_contract({"cluster": "layout"})
        if not apply_result.ok or apply_result.metadata.get("target_test") != "smoke_test_status_server_layout":
            raise SystemExit("failure_apply_contract should select the repeated layout cluster.")
        if apply_result.metadata.get("target_file") != "jarvis_v2/scripts/smoke_test_status_server.py":
            raise SystemExit("failure_apply_contract should choose the dashboard smoke test file.")
        if not apply_result.metadata.get("apply_ready"):
            raise SystemExit("failure_apply_contract should mark repeated clusters apply-ready.")
        if apply_result.metadata.get("apply_contract_sha256") != apply_contract_sha256:
            raise SystemExit(f"failure_apply_contract missed stable contract hash: {apply_result.metadata}")
        assert_exact_test_contract(apply_result.metadata, "failure_apply_contract")
        if apply_result.metadata.get("writes_files") or apply_result.metadata.get("queues_approval") or apply_result.metadata.get("executes_tools"):
            raise SystemExit("failure_apply_contract should stay read-only.")

        cockpit_result = failure_learning_cockpit({"cluster": "layout"})
        if not cockpit_result.ok or cockpit_result.metadata.get("target_test") != "smoke_test_status_server_layout":
            raise SystemExit("failure_learning_cockpit should select the repeated layout cluster.")
        if cockpit_result.metadata.get("cockpit_state") != "FAILURE_LEARNING_READY_FOR_PATCH_REVIEW":
            raise SystemExit(f"failure_learning_cockpit should be ready for patch review: {cockpit_result.metadata}")
        if cockpit_result.metadata.get("target_file") != "jarvis_v2/scripts/smoke_test_status_server.py":
            raise SystemExit("failure_learning_cockpit should carry the target smoke file.")
        if not cockpit_result.metadata.get("focused_command") or "smoke_test_status_server" not in cockpit_result.metadata.get("focused_command"):
            raise SystemExit(f"failure_learning_cockpit missed focused verification command: {cockpit_result.metadata}")
        if cockpit_result.metadata.get("apply_contract_sha256") != apply_contract_sha256:
            raise SystemExit(f"failure_learning_cockpit missed apply contract hash: {cockpit_result.metadata}")
        assert_exact_test_contract(cockpit_result.metadata, "failure_learning_cockpit")
        proof_queue = cockpit_result.metadata.get("failure_proof_queue") or []
        if not proof_queue or cockpit_result.metadata.get("failure_proof_queue_count") != len(proof_queue):
            raise SystemExit(f"failure_learning_cockpit missed proof queue metadata: {cockpit_result.metadata}")
        for expected in ["failure clusters", "failure promotion packet", "failure implementation packet", "failure apply contract", "compileall"]:
            if not any(expected in command for command in proof_queue):
                raise SystemExit(f"failure_learning_cockpit proof queue missed {expected}: {cockpit_result.metadata}")
        assert_command_first_proof_queue(
            cockpit_result,
            "failure_learning_cockpit",
            queue_key="failure_proof_queue",
            proof_key="failure_next_proof_command",
        )
        if cockpit_result.metadata.get("writes_files") or cockpit_result.metadata.get("queues_approval") or cockpit_result.metadata.get("executes_tools"):
            raise SystemExit("failure_learning_cockpit should stay read-only.")

        receipt_result = failure_patch_receipt_packet(
            {
                "summary": ready_patch_summary
            }
        )
        if not receipt_result.ok or receipt_result.metadata.get("target_test") != "smoke_test_status_server_layout":
            raise SystemExit(f"failure_patch_receipt_packet should select the repeated layout cluster: {receipt_result.metadata}")
        if receipt_result.metadata.get("patch_receipt_ready") is not True or receipt_result.metadata.get("completion_review_ready") is not True:
            raise SystemExit(f"failure_patch_receipt_packet should mark complete evidence ready: {receipt_result.metadata}")
        if receipt_result.metadata.get("target_file_in_changed_files") is not True:
            raise SystemExit(f"failure_patch_receipt_packet should require target test file evidence: {receipt_result.metadata}")
        if receipt_result.metadata.get("pre_patch_failing_test_receipt_present") is not True or receipt_result.metadata.get("test_first_receipt_required") is not True:
            raise SystemExit(f"failure_patch_receipt_packet should require test-first failing proof: {receipt_result.metadata}")
        if receipt_result.metadata.get("patch_receipt_hash_present") is not True or receipt_result.metadata.get("patch_receipt_sha256") != patch_receipt_sha256:
            raise SystemExit(f"failure_patch_receipt_packet missed patch receipt hash proof: {receipt_result.metadata}")
        assert_patch_review_contract(receipt_result.metadata, "failure_patch_receipt_packet")
        assert_exact_test_contract(receipt_result.metadata, "failure_patch_receipt_packet")
        if receipt_result.metadata.get("writes_files") or receipt_result.metadata.get("queues_approval") or receipt_result.metadata.get("executes_tools"):
            raise SystemExit("failure_patch_receipt_packet should stay read-only.")

        path_receipt_result = failure_patch_receipt_packet(
            {
                "summary": (
                    "layout; "
                    "changed files /\x55sers/example/Desktop/Claude code/AI agents/jarvis-v2/jarvis_v2/scripts/smoke_test_status_server.py; "
                    "test first /private/tmp/prepatch.log failing assertion reviewed; "
                    "verification python3 -m jarvis_v2.scripts.smoke_test_status_server passed with log /tmp/status.log; "
                    "compile python3 -m py_compile jarvis_v2/tools/feedback.py passed from /var/folders/zc/compile.log; "
                    "rollback revert /\x55sers/example/private/rollback.md; "
                    "notes copied from /\x55sers/example/private/notes.md; "
                    f"patch receipt sha256 {patch_receipt_sha256}"
                )
            }
        )
        if not path_receipt_result.ok or path_receipt_result.metadata.get("patch_receipt_ready") is not True:
            raise SystemExit(f"path-seeded failure_patch_receipt_packet should stay ready: {path_receipt_result.metadata}")
        if path_receipt_result.metadata.get("target_file_in_changed_files") is not True:
            raise SystemExit(f"path-seeded receipt should validate target file from raw evidence: {path_receipt_result.metadata}")
        path_receipt_combined = path_receipt_result.output + str(path_receipt_result.metadata)
        if "<local-path>" not in path_receipt_combined:
            raise SystemExit(f"path-seeded receipt should include redacted local path markers: {path_receipt_combined}")
        assert_no_local_path(path_receipt_combined, "path-seeded failure_patch_receipt_packet")
        if path_receipt_result.metadata.get("writes_files") or path_receipt_result.metadata.get("queues_approval") or path_receipt_result.metadata.get("executes_tools"):
            raise SystemExit("path-seeded failure_patch_receipt_packet should stay read-only.")

        application_bridge_result = failure_patch_application_bridge(
            {
                "summary": ready_patch_summary
            }
        )
        if not application_bridge_result.ok or application_bridge_result.metadata.get("target_test") != "smoke_test_status_server_layout":
            raise SystemExit(f"failure_patch_application_bridge should select the repeated layout cluster: {application_bridge_result.metadata}")
        if application_bridge_result.metadata.get("application_bridge_ready") is not True:
            raise SystemExit(f"failure_patch_application_bridge should mark complete bridge evidence ready: {application_bridge_result.metadata}")
        for key in ["apply_contract_reviewed", "applied_patch_bound_to_contract", "patch_receipt_ready", "pre_patch_failing_test_receipt_present", "focused_verification_present", "compile_pass_present", "rollback_note_present", "patch_receipt_hash_present"]:
            if application_bridge_result.metadata.get(key) is not True:
                raise SystemExit(f"failure_patch_application_bridge missed proof {key}: {application_bridge_result.metadata}")
        if application_bridge_result.metadata.get("apply_contract_hash_matches_expected") is not True:
            raise SystemExit(f"failure_patch_application_bridge missed apply contract hash binding: {application_bridge_result.metadata}")
        if application_bridge_result.metadata.get("apply_contract_sha256") != apply_contract_sha256 or application_bridge_result.metadata.get("supplied_apply_contract_sha256") != apply_contract_sha256:
            raise SystemExit(f"failure_patch_application_bridge contract hashes diverged: {application_bridge_result.metadata}")
        assert_patch_review_contract(application_bridge_result.metadata, "failure_patch_application_bridge")
        assert_exact_test_contract(application_bridge_result.metadata, "failure_patch_application_bridge")
        assert_command_first_proof_queue(application_bridge_result, "failure_patch_application_bridge")
        if application_bridge_result.metadata.get("writes_files") or application_bridge_result.metadata.get("queues_approval") or application_bridge_result.metadata.get("executes_tools"):
            raise SystemExit("failure_patch_application_bridge should stay read-only.")

        path_bridge_result = failure_patch_application_bridge(
            {
                "summary": ready_patch_summary,
                "contract": "/private/tmp/apply-contract.md reviewed",
                "application_bridge": "applied patch bound to contract from /\x55sers/example/private/bridge.md",
            }
        )
        if not path_bridge_result.ok or path_bridge_result.metadata.get("application_bridge_ready") is not True:
            raise SystemExit(f"path-seeded failure_patch_application_bridge should stay ready: {path_bridge_result.metadata}")
        if path_bridge_result.metadata.get("apply_contract_reviewed") is not True or path_bridge_result.metadata.get("applied_patch_bound_to_contract") is not True:
            raise SystemExit(f"path-seeded bridge should validate raw contract/application evidence: {path_bridge_result.metadata}")
        path_bridge_combined = path_bridge_result.output + str(path_bridge_result.metadata)
        if "<local-path>" not in path_bridge_combined:
            raise SystemExit(f"path-seeded bridge should include redacted local path markers: {path_bridge_combined}")
        assert_no_local_path(path_bridge_combined, "path-seeded failure_patch_application_bridge")
        if path_bridge_result.metadata.get("writes_files") or path_bridge_result.metadata.get("queues_approval") or path_bridge_result.metadata.get("executes_tools"):
            raise SystemExit("path-seeded failure_patch_application_bridge should stay read-only.")

        completion_gate = failure_patch_completion_gate(
            {
                "summary": ready_patch_summary
            }
        )
        if not completion_gate.ok or completion_gate.metadata.get("target_test") != "smoke_test_status_server_layout":
            raise SystemExit(f"failure_patch_completion_gate should select the repeated layout cluster: {completion_gate.metadata}")
        if completion_gate.metadata.get("gate_state") != "FAILURE_PATCH_COMPLETION_READY_FOR_CLAIM_REVIEW":
            raise SystemExit(f"failure_patch_completion_gate should be ready for claim review: {completion_gate.metadata}")
        if completion_gate.metadata.get("completion_claim_ready") is not True or completion_gate.metadata.get("apply_contract_reviewed") is not True:
            raise SystemExit(f"failure_patch_completion_gate missed ready contract metadata: {completion_gate.metadata}")
        if completion_gate.metadata.get("application_bridge_ready") is not True or completion_gate.metadata.get("applied_patch_bound_to_contract") is not True:
            raise SystemExit(f"failure_patch_completion_gate missed bridge metadata: {completion_gate.metadata}")
        if completion_gate.metadata.get("apply_contract_hash_matches_expected") is not True:
            raise SystemExit(f"failure_patch_completion_gate missed apply contract hash binding: {completion_gate.metadata}")
        if completion_gate.metadata.get("pre_patch_failing_test_receipt_present") is not True:
            raise SystemExit(f"failure_patch_completion_gate missed test-first receipt metadata: {completion_gate.metadata}")
        if completion_gate.metadata.get("patch_receipt_hash_present") is not True or completion_gate.metadata.get("patch_receipt_sha256") != patch_receipt_sha256:
            raise SystemExit(f"failure_patch_completion_gate missed patch receipt hash proof: {completion_gate.metadata}")
        if "completion claim gate" not in " ".join(completion_gate.metadata.get("proof_queue", [])):
            raise SystemExit(f"failure_patch_completion_gate missed completion proof queue: {completion_gate.metadata}")
        assert_patch_review_contract(completion_gate.metadata, "failure_patch_completion_gate")
        assert_exact_test_contract(completion_gate.metadata, "failure_patch_completion_gate")
        assert_command_first_proof_queue(completion_gate, "failure_patch_completion_gate")
        if completion_gate.metadata.get("writes_files") or completion_gate.metadata.get("queues_approval") or completion_gate.metadata.get("executes_tools"):
            raise SystemExit("failure_patch_completion_gate should stay read-only.")

        path_completion_gate = failure_patch_completion_gate(
            {
                "summary": ready_patch_summary,
                "contract": "/private/tmp/apply-contract.md reviewed",
            }
        )
        if not path_completion_gate.ok or path_completion_gate.metadata.get("gate_state") != "FAILURE_PATCH_COMPLETION_READY_FOR_CLAIM_REVIEW":
            raise SystemExit(f"path-seeded failure_patch_completion_gate should stay ready: {path_completion_gate.metadata}")
        if path_completion_gate.metadata.get("apply_contract_reviewed") is not True:
            raise SystemExit(f"path-seeded completion gate should validate raw contract evidence: {path_completion_gate.metadata}")
        path_completion_combined = path_completion_gate.output + str(path_completion_gate.metadata)
        if "<local-path>" not in path_completion_combined:
            raise SystemExit(f"path-seeded completion gate should include redacted local path markers: {path_completion_combined}")
        assert_no_local_path(path_completion_combined, "path-seeded failure_patch_completion_gate")
        if path_completion_gate.metadata.get("writes_files") or path_completion_gate.metadata.get("queues_approval") or path_completion_gate.metadata.get("executes_tools"):
            raise SystemExit("path-seeded failure_patch_completion_gate should stay read-only.")

        handoff_result = failure_patch_handoff_packet(
            {
                "summary": ready_patch_summary
            }
        )
        if not handoff_result.ok or handoff_result.metadata.get("target_test") != "smoke_test_status_server_layout":
            raise SystemExit(f"failure_patch_handoff_packet should select the repeated layout cluster: {handoff_result.metadata}")
        if handoff_result.metadata.get("handoff_state") != "FAILURE_PATCH_HANDOFF_READY_FOR_COMPLETION_REVIEW":
            raise SystemExit(f"failure_patch_handoff_packet should be ready for completion review: {handoff_result.metadata}")
        if handoff_result.metadata.get("ready_for_completion_review") is not True or handoff_result.metadata.get("patch_receipt_ready") is not True:
            raise SystemExit(f"failure_patch_handoff_packet missed ready handoff metadata: {handoff_result.metadata}")
        if handoff_result.metadata.get("pre_patch_failing_test_receipt_present") is not True:
            raise SystemExit(f"failure_patch_handoff_packet missed test-first receipt metadata: {handoff_result.metadata}")
        if handoff_result.metadata.get("application_bridge_ready") is not True or handoff_result.metadata.get("applied_patch_bound_to_contract") is not True:
            raise SystemExit(f"failure_patch_handoff_packet missed bridge metadata: {handoff_result.metadata}")
        if handoff_result.metadata.get("apply_contract_hash_matches_expected") is not True:
            raise SystemExit(f"failure_patch_handoff_packet missed apply contract hash binding: {handoff_result.metadata}")
        if handoff_result.metadata.get("patch_receipt_hash_present") is not True or handoff_result.metadata.get("patch_receipt_sha256") != patch_receipt_sha256:
            raise SystemExit(f"failure_patch_handoff_packet missed patch receipt hash proof: {handoff_result.metadata}")
        if "evidence ledger" not in " ".join(handoff_result.metadata.get("post_claim_commands", [])):
            raise SystemExit(f"failure_patch_handoff_packet missed post-claim review queue: {handoff_result.metadata}")
        assert_patch_review_contract(handoff_result.metadata, "failure_patch_handoff_packet")
        assert_exact_test_contract(handoff_result.metadata, "failure_patch_handoff_packet")
        if not _failure_patch_handoff_ready(handoff_result.metadata):
            raise SystemExit(f"failure_patch_handoff_packet should pass the production handoff validator: {handoff_result.metadata}")
        for key, value in [
            ("ready_for_completion_review", False),
            ("handoff_state", "FAILURE_PATCH_HANDOFF_HELD"),
            ("apply_contract_hash_matches_expected", False),
            ("exact_test_contract_ready", False),
            ("application_bridge_ready", False),
            ("pre_patch_failing_test_receipt_present", False),
            ("review_token_authorizes_patch_application", True),
            ("authorizes_completion_claim", True),
            ("failure_patch_review_token_sha256", "0" * 64),
        ]:
            tampered = dict(handoff_result.metadata)
            tampered[key] = value
            if _failure_patch_handoff_ready(tampered):
                raise SystemExit(f"failure_patch_handoff_packet validator accepted tampered {key}: {tampered}")
        tampered_rows = [dict(row) for row in handoff_result.metadata.get("patch_review_contract_rows") or []]
        tampered_rows[0]["authorizes_file_write"] = True
        tampered = dict(handoff_result.metadata)
        tampered["patch_review_contract_rows"] = tampered_rows
        if _failure_patch_handoff_ready(tampered):
            raise SystemExit(f"failure_patch_handoff_packet validator accepted authority-bearing review rows: {tampered}")
        if handoff_result.metadata.get("writes_files") or handoff_result.metadata.get("queues_approval") or handoff_result.metadata.get("executes_tools"):
            raise SystemExit("failure_patch_handoff_packet should stay read-only.")

        closeout_result = failure_patch_closeout_packet(
            {
                "summary": ready_learning_summary
            }
        )
        if not closeout_result.ok or closeout_result.metadata.get("target_test") != "smoke_test_status_server_layout":
            raise SystemExit(f"failure_patch_closeout_packet should select the repeated layout cluster: {closeout_result.metadata}")
        if closeout_result.metadata.get("closeout_state") != "FAILURE_PATCH_CLOSEOUT_READY_FOR_LEARNING_RECORD":
            raise SystemExit(f"failure_patch_closeout_packet should be ready for learning record: {closeout_result.metadata}")
        if closeout_result.metadata.get("ready_for_learning_record") is not True:
            raise SystemExit(f"failure_patch_closeout_packet missed ready closeout metadata: {closeout_result.metadata}")
        for key in ["completion_audit_reviewed", "evidence_ledger_reviewed", "completion_claim_gate_reviewed", "post_claim_reviewed", "application_bridge_ready", "applied_patch_bound_to_contract", "pre_patch_failing_test_receipt_present"]:
            if closeout_result.metadata.get(key) is not True:
                raise SystemExit(f"failure_patch_closeout_packet missed closeout proof {key}: {closeout_result.metadata}")
        if closeout_result.metadata.get("apply_contract_hash_matches_expected") is not True:
            raise SystemExit(f"failure_patch_closeout_packet missed apply contract hash binding: {closeout_result.metadata}")
        if closeout_result.metadata.get("patch_receipt_hash_present") is not True or closeout_result.metadata.get("patch_receipt_sha256") != patch_receipt_sha256:
            raise SystemExit(f"failure_patch_closeout_packet missed patch receipt hash proof: {closeout_result.metadata}")
        if "learning review" not in " ".join(closeout_result.metadata.get("learning_record_commands", [])):
            raise SystemExit(f"failure_patch_closeout_packet missed learning record queue: {closeout_result.metadata}")
        assert_patch_review_contract(closeout_result.metadata, "failure_patch_closeout_packet")
        assert_exact_test_contract(closeout_result.metadata, "failure_patch_closeout_packet")
        if closeout_result.metadata.get("writes_files") or closeout_result.metadata.get("queues_approval") or closeout_result.metadata.get("executes_tools"):
            raise SystemExit("failure_patch_closeout_packet should stay read-only.")

        learning_record_result = failure_learning_record_packet(
            {
                "summary": ready_learning_summary
            }
        )
        if not learning_record_result.ok or learning_record_result.metadata.get("target_test") != "smoke_test_status_server_layout":
            raise SystemExit(f"failure_learning_record_packet should select the repeated layout cluster: {learning_record_result.metadata}")
        if learning_record_result.metadata.get("record_state") != "FAILURE_LEARNING_RECORD_READY":
            raise SystemExit(f"failure_learning_record_packet should be ready: {learning_record_result.metadata}")
        for key in ["closeout_ready", "after_action_learning_reviewed", "learning_record_target_present", "regression_test_linked", "durable_learning_record", "application_bridge_ready", "applied_patch_bound_to_contract", "pre_patch_failing_test_receipt_present", "patch_receipt_hash_present", "learning_record_hash_present", "learning_artifact_hashes_present"]:
            if learning_record_result.metadata.get(key) is not True:
                raise SystemExit(f"failure_learning_record_packet missed learning proof {key}: {learning_record_result.metadata}")
        if learning_record_result.metadata.get("apply_contract_hash_matches_expected") is not True:
            raise SystemExit(f"failure_learning_record_packet missed apply contract hash binding: {learning_record_result.metadata}")
        if learning_record_result.metadata.get("patch_receipt_sha256") != patch_receipt_sha256 or learning_record_result.metadata.get("learning_record_sha256") != learning_record_sha256:
            raise SystemExit(f"failure_learning_record_packet hash values diverged: {learning_record_result.metadata}")
        assert_exact_test_contract(learning_record_result.metadata, "failure_learning_record_packet")
        assert_failure_learning_record_token(learning_record_result.metadata, "failure_learning_record_packet")
        assert_patch_review_contract(learning_record_result.metadata, "failure_learning_record_packet")
        if learning_record_result.metadata.get("writes_files") or learning_record_result.metadata.get("queues_approval") or learning_record_result.metadata.get("executes_tools"):
            raise SystemExit("failure_learning_record_packet should stay read-only.")

        path_learning_record = failure_learning_record_packet(
            {
                "summary": ready_learning_summary,
                "after_action_learning": "/\x55sers/example/Jarvis/after-action.md reviewed and saved",
                "record_target": "/private/tmp/jarvis-learning-review.md learning review",
                "regression_link": "/var/folders/jarvis/status smoke_test_status_server_layout regression linked",
                "durability": "/tmp/jarvis-learning-checkpoint.md checkpoint saved",
            }
        )
        if not path_learning_record.ok or path_learning_record.metadata.get("record_state") != "FAILURE_LEARNING_RECORD_READY":
            raise SystemExit(f"path-seeded failure_learning_record_packet should stay ready: {path_learning_record.metadata}")
        for key in ["after_action_learning_reviewed", "learning_record_target_present", "regression_test_linked", "durable_learning_record"]:
            if path_learning_record.metadata.get(key) is not True:
                raise SystemExit(f"path-seeded failure_learning_record_packet should validate raw {key}: {path_learning_record.metadata}")
        path_learning_record_combined = path_learning_record.output + str(path_learning_record.metadata)
        if "<local-path>" not in path_learning_record_combined:
            raise SystemExit(f"path-seeded failure_learning_record_packet should include redacted local path markers: {path_learning_record_combined}")
        assert_no_local_path(path_learning_record_combined, "path-seeded failure_learning_record_packet")
        assert_failure_learning_record_token(path_learning_record.metadata, "path-seeded failure_learning_record_packet")
        if path_learning_record.metadata.get("writes_files") or path_learning_record.metadata.get("queues_approval") or path_learning_record.metadata.get("executes_tools"):
            raise SystemExit("path-seeded failure_learning_record_packet should stay read-only.")

        closure_ledger_result = failure_learning_closure_ledger(
            {
                "summary": ready_learning_summary
            }
        )
        if not closure_ledger_result.ok or closure_ledger_result.metadata.get("target_test") != "smoke_test_status_server_layout":
            raise SystemExit(f"failure_learning_closure_ledger should select the repeated layout cluster: {closure_ledger_result.metadata}")
        if closure_ledger_result.metadata.get("ledger_state") != "FAILURE_LEARNING_CLOSURE_READY":
            raise SystemExit(f"failure_learning_closure_ledger should be ready: {closure_ledger_result.metadata}")
        if closure_ledger_result.metadata.get("ready_as_durable_learning_closure") is not True:
            raise SystemExit(f"failure_learning_closure_ledger missed durable closure metadata: {closure_ledger_result.metadata}")
        stage_rows = closure_ledger_result.metadata.get("stage_rows") or []
        proof_queue = closure_ledger_result.metadata.get("proof_queue") or []
        if closure_ledger_result.metadata.get("stage_count") != len(stage_rows) or len(stage_rows) < 7:
            raise SystemExit(f"failure_learning_closure_ledger missed stage rows: {closure_ledger_result.metadata}")
        if not all(row.get("ready") for row in stage_rows):
            raise SystemExit(f"failure_learning_closure_ledger should mark all closure stages ready: {closure_ledger_result.metadata}")
        if closure_ledger_result.metadata.get("proof_queue_count") != len(proof_queue) or "learning review" not in proof_queue:
            raise SystemExit(f"failure_learning_closure_ledger missed proof queue: {closure_ledger_result.metadata}")
        assert_command_first_proof_queue(closure_ledger_result, "failure_learning_closure_ledger")
        if closure_ledger_result.metadata.get("learning_artifact_hashes_present") is not True:
            raise SystemExit(f"failure_learning_closure_ledger missed learning artifact hashes: {closure_ledger_result.metadata}")
        if closure_ledger_result.metadata.get("apply_contract_hash_matches_expected") is not True:
            raise SystemExit(f"failure_learning_closure_ledger missed apply contract hash binding: {closure_ledger_result.metadata}")
        if closure_ledger_result.metadata.get("patch_receipt_sha256") != patch_receipt_sha256 or closure_ledger_result.metadata.get("learning_record_sha256") != learning_record_sha256:
            raise SystemExit(f"failure_learning_closure_ledger hash values diverged: {closure_ledger_result.metadata}")
        assert_exact_test_contract(closure_ledger_result.metadata, "failure_learning_closure_ledger")
        assert_failure_learning_record_token(closure_ledger_result.metadata, "failure_learning_closure_ledger")
        if closure_ledger_result.metadata.get("failure_learning_record_token_sha256") != learning_record_result.metadata.get("failure_learning_record_token_sha256"):
            raise SystemExit(f"failure_learning_closure_ledger did not carry the learning record token forward: {closure_ledger_result.metadata}")
        assert_failure_learning_closure_token_boundary(closure_ledger_result.metadata, "failure_learning_closure_ledger")
        if closure_ledger_result.metadata.get("next_patch_requires_fresh_learning_closure") is not True:
            raise SystemExit(f"failure_learning_closure_ledger missed fresh next-patch closure requirement: {closure_ledger_result.metadata}")
        if closure_ledger_result.metadata.get("next_completion_claim_requires_fresh_learning_closure") is not True:
            raise SystemExit(f"failure_learning_closure_ledger missed fresh completion-claim closure requirement: {closure_ledger_result.metadata}")
        if "prior learning closure token is proof-only" not in closure_ledger_result.output or "closure token boundary rows: 3" not in closure_ledger_result.output:
            raise SystemExit("failure_learning_closure_ledger missed proof-only closure token output.")
        assert_patch_review_contract(closure_ledger_result.metadata, "failure_learning_closure_ledger")

        path_ready_learning_summary = (
            ready_patch_summary
            + "; completion audit reviewed; "
            "evidence ledger reviewed; "
            "completion claim gate reviewed; "
            "review post-claim review complete; "
            "after-action learning /\x55sers/example/Jarvis/after-action.md reviewed and saved; "
            "record target /private/tmp/jarvis-learning-review.md learning review; "
            "regression /var/folders/jarvis/status smoke_test_status_server_layout regression linked; "
            "durability /tmp/jarvis-learning-checkpoint.md checkpoint saved; "
            f"learning record sha256 {learning_record_sha256}"
        )
        path_closure_ledger = failure_learning_closure_ledger(
            {
                "summary": path_ready_learning_summary
            }
        )
        if not path_closure_ledger.ok or path_closure_ledger.metadata.get("ledger_state") != "FAILURE_LEARNING_CLOSURE_READY":
            raise SystemExit(f"path-seeded failure_learning_closure_ledger should stay ready: {path_closure_ledger.metadata}")
        for key in ["ready_as_durable_learning_closure", "failure_learning_record_token_present", "failure_learning_closure_token_present", "failure_learning_closure_token_boundary_ready"]:
            if path_closure_ledger.metadata.get(key) is not True:
                raise SystemExit(f"path-seeded failure_learning_closure_ledger missed closure proof {key}: {path_closure_ledger.metadata}")
        path_closure_combined = path_closure_ledger.output + str(path_closure_ledger.metadata)
        if "<local-path>" not in path_closure_combined:
            raise SystemExit(f"path-seeded failure_learning_closure_ledger should include redacted local path markers: {path_closure_combined}")
        assert_no_local_path(path_closure_combined, "path-seeded failure_learning_closure_ledger")
        assert_failure_learning_record_token(path_closure_ledger.metadata, "path-seeded failure_learning_closure_ledger")
        assert_failure_learning_closure_token_boundary(path_closure_ledger.metadata, "path-seeded failure_learning_closure_ledger")
        if path_closure_ledger.metadata.get("writes_files") or path_closure_ledger.metadata.get("queues_approval") or path_closure_ledger.metadata.get("executes_tools"):
            raise SystemExit("path-seeded failure_learning_closure_ledger should stay read-only.")

        carried_patch_review_tokens = {
            receipt_result.metadata.get("failure_patch_review_token_sha256"),
            application_bridge_result.metadata.get("failure_patch_review_token_sha256"),
            completion_gate.metadata.get("failure_patch_review_token_sha256"),
            handoff_result.metadata.get("failure_patch_review_token_sha256"),
            closeout_result.metadata.get("failure_patch_review_token_sha256"),
            learning_record_result.metadata.get("failure_patch_review_token_sha256"),
            closure_ledger_result.metadata.get("failure_patch_review_token_sha256"),
        }
        if len(carried_patch_review_tokens) != 1:
            raise SystemExit(f"patch review token should carry unchanged across the learning chain: {carried_patch_review_tokens}")
        if closure_ledger_result.metadata.get("writes_files") or closure_ledger_result.metadata.get("queues_approval") or closure_ledger_result.metadata.get("executes_tools"):
            raise SystemExit("failure_learning_closure_ledger should stay read-only.")

        incomplete_receipt = failure_patch_receipt_packet({"summary": "layout; verification not yet"})
        if not incomplete_receipt.ok or incomplete_receipt.metadata.get("patch_receipt_ready") is not False:
            raise SystemExit(f"failure_patch_receipt_packet should hold incomplete receipts: {incomplete_receipt.metadata}")
        for expected_missing in ["changed_files", "pre_patch_failing_test_receipt", "compile_pass", "rollback_note", "patch_receipt_sha256"]:
            if expected_missing not in incomplete_receipt.metadata.get("missing", []):
                raise SystemExit(f"failure_patch_receipt_packet missed incomplete receipt gap {expected_missing}: {incomplete_receipt.metadata}")

        incomplete_application_bridge = failure_patch_application_bridge(
            {
                "summary": (
                    "layout; changed files jarvis_v2/scripts/smoke_test_status_server.py; "
                    "test first pre-patch failing assertion reviewed; "
                    "verification python3 -m jarvis_v2.scripts.smoke_test_status_server passed; "
                    "compile python3 -m compileall -q jarvis_v2 passed; "
                    "rollback scoped assertion can be reverted; "
                    "contract apply contract reviewed"
                )
            }
        )
        if not incomplete_application_bridge.ok or incomplete_application_bridge.metadata.get("application_bridge_ready") is not False:
            raise SystemExit(f"failure_patch_application_bridge should hold missing applied-patch bridge proof: {incomplete_application_bridge.metadata}")
        if "applied_patch_bound_to_contract" not in incomplete_application_bridge.metadata.get("missing", []):
            raise SystemExit(f"failure_patch_application_bridge missed bridge gap: {incomplete_application_bridge.metadata}")
        if "apply_contract_sha256" not in incomplete_application_bridge.metadata.get("missing", []):
            raise SystemExit(f"failure_patch_application_bridge missed apply contract hash gap: {incomplete_application_bridge.metadata}")

        mismatched_application_bridge = failure_patch_application_bridge(
            {
                "summary": ready_patch_summary.replace(apply_contract_sha256, "6" * 64)
            }
        )
        if not mismatched_application_bridge.ok or mismatched_application_bridge.metadata.get("application_bridge_ready") is not False:
            raise SystemExit(f"failure_patch_application_bridge should hold mismatched contract hashes: {mismatched_application_bridge.metadata}")
        if mismatched_application_bridge.metadata.get("apply_contract_hash_matches_expected") is not False:
            raise SystemExit(f"failure_patch_application_bridge should reject mismatched contract hash: {mismatched_application_bridge.metadata}")
        if "apply_contract_sha256_matches_reviewed_contract" not in mismatched_application_bridge.metadata.get("missing", []):
            raise SystemExit(f"failure_patch_application_bridge missed mismatched contract hash gap: {mismatched_application_bridge.metadata}")

        incomplete_completion_gate = failure_patch_completion_gate({"summary": "layout; verification not yet"})
        if not incomplete_completion_gate.ok or incomplete_completion_gate.metadata.get("completion_claim_ready") is not False:
            raise SystemExit(f"failure_patch_completion_gate should hold incomplete evidence: {incomplete_completion_gate.metadata}")
        for expected_missing in ["changed_files", "pre_patch_failing_test_receipt", "compile_pass", "rollback_note", "apply_contract_reviewed", "failure_patch_application_bridge_ready", "applied_patch_bound_to_contract", "apply_contract_sha256_matches_reviewed_contract", "patch_receipt_ready", "patch_receipt_sha256"]:
            if expected_missing not in incomplete_completion_gate.metadata.get("missing", []):
                raise SystemExit(f"failure_patch_completion_gate missed incomplete gate gap {expected_missing}: {incomplete_completion_gate.metadata}")

        incomplete_closeout = failure_patch_closeout_packet({"summary": "layout; verification not yet"})
        if not incomplete_closeout.ok or incomplete_closeout.metadata.get("ready_for_learning_record") is not False:
            raise SystemExit(f"failure_patch_closeout_packet should hold incomplete closeout proof: {incomplete_closeout.metadata}")
        for expected_missing in ["completion_audit_reviewed", "evidence_ledger_reviewed", "completion_claim_gate_reviewed", "post_claim_reviewed", "patch_receipt_sha256"]:
            if expected_missing not in incomplete_closeout.metadata.get("missing", []):
                raise SystemExit(f"failure_patch_closeout_packet missed incomplete closeout gap {expected_missing}: {incomplete_closeout.metadata}")

        incomplete_learning_record = failure_learning_record_packet({"summary": "layout; verification not yet"})
        if not incomplete_learning_record.ok or incomplete_learning_record.metadata.get("ready_for_durable_learning_record") is not False:
            raise SystemExit(f"failure_learning_record_packet should hold incomplete learning proof: {incomplete_learning_record.metadata}")
        for expected_missing in ["after_action_learning_reviewed", "learning_record_target", "regression_test_linked", "durable_learning_record", "patch_receipt_sha256", "learning_record_sha256"]:
            if expected_missing not in incomplete_learning_record.metadata.get("missing", []):
                raise SystemExit(f"failure_learning_record_packet missed incomplete learning gap {expected_missing}: {incomplete_learning_record.metadata}")

        incomplete_closure_ledger = failure_learning_closure_ledger({"summary": "layout; verification not yet"})
        if not incomplete_closure_ledger.ok or incomplete_closure_ledger.metadata.get("ready_as_durable_learning_closure") is not False:
            raise SystemExit(f"failure_learning_closure_ledger should hold incomplete closure proof: {incomplete_closure_ledger.metadata}")
        for expected_missing in ["learning_record", "after_action_learning_reviewed", "durable_learning_record", "learning_artifact_hashes"]:
            if expected_missing not in incomplete_closure_ledger.metadata.get("missing", []):
                raise SystemExit(f"failure_learning_closure_ledger missed incomplete closure gap {expected_missing}: {incomplete_closure_ledger.metadata}")

        missing_promotion = failure_promotion_packet({"cluster": "calendar"})
        if missing_promotion.ok or missing_promotion.metadata.get("promoted"):
            raise SystemExit("failure_promotion_packet should reject missing selectors safely.")

        missing_implementation = failure_implementation_packet({"cluster": "calendar"})
        if missing_implementation.ok or missing_implementation.metadata.get("implementation_ready"):
            raise SystemExit("failure_implementation_packet should reject missing selectors safely.")

        missing_apply = failure_apply_contract({"cluster": "calendar"})
        if missing_apply.ok or missing_apply.metadata.get("apply_ready"):
            raise SystemExit("failure_apply_contract should reject missing selectors safely.")

        missing_cockpit = failure_learning_cockpit({"cluster": "calendar"})
        if missing_cockpit.metadata.get("cockpit_state") != "FAILURE_LEARNING_HELD" or missing_cockpit.metadata.get("ready_for_patch_review"):
            raise SystemExit(f"failure_learning_cockpit should hold missing selectors safely: {missing_cockpit.metadata}")
        for label, result in [
            ("failure_patch_receipt_packet incomplete", incomplete_receipt),
            ("failure_patch_application_bridge incomplete", incomplete_application_bridge),
            ("failure_patch_application_bridge mismatched", mismatched_application_bridge),
            ("failure_patch_completion_gate incomplete", incomplete_completion_gate),
            ("failure_patch_closeout_packet incomplete", incomplete_closeout),
            ("failure_learning_record_packet incomplete", incomplete_learning_record),
            ("failure_learning_closure_ledger incomplete", incomplete_closure_ledger),
            ("failure_promotion_packet missing", missing_promotion),
            ("failure_implementation_packet missing", missing_implementation),
            ("failure_apply_contract missing", missing_apply),
            ("failure_learning_cockpit missing", missing_cockpit),
        ]:
            assert_feedback_result_no_future_authority(result, label)


if __name__ == "__main__":
    main()
