from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from jarvis_v2.agent.receipts import runtime_result_receipt
from jarvis_v2.agent.types import Plan, RuntimeResult, ToolResult
from jarvis_v2.scripts.test_runtime import handle_runtime_case, make_temp_runtime
from jarvis_v2.tools.audit import (
    _checkpoint_recovery_execute_receipt_invalid_reason,
    _first_safe_text,
    _ids_from_text,
    _metadata_bool,
    _metadata_int,
    _short,
    _short_raw,
    make_audit_tools,
)


class HostileRow:
    def __init__(self, marker: str) -> None:
        self.marker = marker

    def keys(self):
        raise RuntimeError(self.marker)

    def __getitem__(self, _key):
        raise RuntimeError(self.marker)

    def __str__(self) -> str:
        return self.marker


class HostileMetadataValue:
    def __init__(self, marker: str) -> None:
        self.marker = marker

    def __bool__(self) -> bool:
        raise RuntimeError(self.marker)

    def __str__(self) -> str:
        raise RuntimeError(self.marker)

    def __repr__(self) -> str:
        return f"<hostile-audit-metadata {self.marker}>"


def assert_no_future_authority(metadata: dict, label: str) -> None:
    for key in ["authorizes_execution", "authorizes_completion_claim", "approval_granted"]:
        if metadata.get(key) is not False:
            raise SystemExit(f"{label} should keep {key}=False: {metadata}")


def assert_audit_lookup_recovery(result, label: str, expected_commands: list[str]) -> None:
    if result.ok:
        raise SystemExit(f"{label} should fail closed: {result}")
    for command in expected_commands:
        if command not in result.output:
            raise SystemExit(f"{label} missed recovery command {command!r}: {result.output}")
    if result.metadata.get("next_command") != expected_commands[0]:
        raise SystemExit(f"{label} missed first recovery command: {result.metadata}")
    if result.metadata.get("recovery_commands") != expected_commands:
        raise SystemExit(f"{label} missed ordered recovery metadata: {result.metadata}")
    if result.metadata.get("retry_requires_audit_refresh") is not True:
        raise SystemExit(f"{label} should require an audit refresh: {result.metadata}")
    if result.metadata.get("authorizes_retry") is not False:
        raise SystemExit(f"{label} should not authorize retry: {result.metadata}")
    assert_no_future_authority(result.metadata, label)


def assert_audit_metadata_bool_is_exact() -> None:
    if _metadata_bool(True) is not True:
        raise SystemExit("audit exact metadata bool rejected True")
    if _metadata_bool(False) is not False:
        raise SystemExit("audit exact metadata bool rejected False")
    for value in ("true", "false", "yes", "no", 1, 0, [True], {"ready": True}, None):
        if _metadata_bool(value) is not False:
            raise SystemExit(f"audit exact metadata bool accepted malformed truthy value: {value!r}")
    if _metadata_bool("false", default=True) is not True:
        raise SystemExit("audit exact metadata bool did not preserve explicit default")


def assert_runtime_result_trace_bools_are_exact(runtime) -> None:
    malformed_trace = runtime._result_trace(
        ToolResult(
            "fake_tool",
            False,
            "synthetic result for runtime trace boolean regression",
            {
                "requires_confirmation": "true",
                "executed_handler": "true",
                "approval_rerun_exact": "true",
                "approval_id": 77,
                "failure_kind": "synthetic",
            },
        )
    )
    for key in ("requires_confirmation", "executed_handler", "approval_rerun_exact"):
        if malformed_trace.get(key) is not False:
            raise SystemExit(f"runtime result trace accepted malformed boolean {key}: {malformed_trace}")
    exact_trace = runtime._result_trace(
        ToolResult(
            "fake_tool",
            True,
            "synthetic exact true trace",
            {
                "requires_confirmation": True,
                "executed_handler": True,
                "approval_rerun_exact": True,
            },
        )
    )
    for key in ("requires_confirmation", "executed_handler", "approval_rerun_exact"):
        if exact_trace.get(key) is not True:
            raise SystemExit(f"runtime result trace rejected exact true boolean {key}: {exact_trace}")


def assert_runtime_trace_handler_execution_truth(runtime) -> None:
    plan = Plan(goal="synthetic handler truth", actions=[])

    def build(results: list[ToolResult]) -> dict:
        return runtime._build_runtime_trace(
            user_input="synthetic handler truth",
            route="tools",
            plan=plan,
            results=results,
            verified=all(result.ok for result in results),
            verification="synthetic verification",
            chat_response={},
            approved=False,
            approved_approval_id=None,
            approval_queue_before=0,
            approval_queue_after=0,
        )

    cases = [
        (
            "explicit false",
            [ToolResult("fake_tool", False, "refused", {"executed_handler": False, "requires_confirmation": False})],
            False,
            0,
            "not_run",
        ),
        (
            "approval held",
            [ToolResult("fake_tool", False, "held", {"executed_handler": False, "requires_confirmation": True})],
            False,
            0,
            "held",
        ),
        (
            "malformed execution flag",
            [ToolResult("fake_tool", False, "malformed", {"executed_handler": "true", "requires_confirmation": False})],
            False,
            0,
            "not_run",
        ),
        (
            "missing execution receipt",
            [ToolResult("fake_tool", False, "missing", {})],
            False,
            0,
            "not_run",
        ),
        (
            "completed",
            [ToolResult("fake_tool", True, "done", {"executed_handler": True, "requires_confirmation": False})],
            True,
            1,
            "completed",
        ),
        (
            "current handler overrides historical target receipt",
            [
                ToolResult(
                    "fake_audit_tool",
                    True,
                    "historical target was held",
                    {"handler_invoked": True, "executed_handler": False, "requires_confirmation": False},
                )
            ],
            True,
            1,
            "completed",
        ),
        (
            "partial",
            [
                ToolResult("fake_tool", True, "done", {"executed_handler": True, "requires_confirmation": False}),
                ToolResult("other_tool", False, "refused", {"executed_handler": False, "requires_confirmation": False}),
            ],
            True,
            1,
            "partial",
        ),
    ]
    for label, results, expected_ran, expected_count, expected_status in cases:
        trace = build(results)
        execution_stage = next(
            (stage for stage in trace.get("stages", []) if stage.get("stage") == "execution"),
            {},
        )
        if trace.get("ran_tool_handlers") is not expected_ran:
            raise SystemExit(f"{label} runtime trace handler truth wrong: {trace}")
        if trace.get("executed_handler_count") != expected_count:
            raise SystemExit(f"{label} runtime trace handler count wrong: {trace}")
        if trace.get("tool_result_count") != len(results):
            raise SystemExit(f"{label} runtime trace result count wrong: {trace}")
        if execution_stage.get("status") != expected_status:
            raise SystemExit(f"{label} runtime trace execution stage wrong: {execution_stage}")


def assert_runtime_result_receipt_bools_are_exact() -> None:
    malformed_result = RuntimeResult(
        user_input="synthetic malformed runtime trace receipt",
        plan=Plan(goal="synthetic receipt regression", actions=[]),
        tool_results=[],
        verified=False,
        response="synthetic response",
        metadata={
            "runtime_route": "tool",
            "runtime_trace": {
                "approved": "true",
                "approval_required": "true",
                "ran_tool_handlers": "true",
                "planner_metadata": {
                    "model_planner_attempted": "true",
                    "model_planner_used": "true",
                    "model_planner_fell_back": "true",
                },
            },
        },
    )
    malformed_receipt = runtime_result_receipt(malformed_result, pending_approvals=0)
    for key in ("approved", "approval_required", "ran_tool_handlers"):
        if malformed_receipt.get(key) is not False:
            raise SystemExit(f"runtime result receipt accepted malformed boolean {key}: {malformed_receipt}")
    for key in ("planner_model_planner_attempted", "planner_model_planner_used", "planner_model_planner_fell_back"):
        if malformed_receipt.get(key) is not False:
            raise SystemExit(f"runtime result receipt accepted malformed planner boolean {key}: {malformed_receipt}")

    exact_result = RuntimeResult(
        user_input="synthetic exact runtime trace receipt",
        plan=Plan(goal="synthetic exact receipt", actions=[]),
        tool_results=[],
        verified=True,
        response="synthetic response",
        metadata={
            "runtime_route": "tool",
            "runtime_trace": {
                "approved": True,
                "approval_required": True,
                "ran_tool_handlers": True,
                "planner_metadata": {
                    "model_planner_attempted": True,
                    "model_planner_used": True,
                    "model_planner_fell_back": True,
                },
            },
        },
    )
    exact_receipt = runtime_result_receipt(exact_result, pending_approvals=1)
    for key in ("approved", "approval_required", "ran_tool_handlers"):
        if exact_receipt.get(key) is not True:
            raise SystemExit(f"runtime result receipt rejected exact boolean {key}: {exact_receipt}")
    for key in ("planner_model_planner_attempted", "planner_model_planner_used", "planner_model_planner_fell_back"):
        if exact_receipt.get(key) is not True:
            raise SystemExit(f"runtime result receipt rejected exact planner boolean {key}: {exact_receipt}")


def assert_audit_formatters_tolerate_hostile_values() -> None:
    marker = "AUDIT_FORMATTER_SHOULD_NOT_LEAK /\x55sers/example/private/audit.sqlite"
    hostile = HostileMetadataValue(marker)
    if _short(hostile) != "":
        raise SystemExit("audit _short should ignore hostile values")
    if _short_raw(hostile) != "":
        raise SystemExit("audit _short_raw should ignore hostile values")
    if _ids_from_text(hostile, ("approval #", "approval id")) != []:
        raise SystemExit("audit _ids_from_text should ignore hostile values")
    if _first_safe_text(hostile, "safe-fallback") != "safe-fallback":
        raise SystemExit("audit _first_safe_text should skip hostile values and preserve safe fallback")
    invalid_reason = _checkpoint_recovery_execute_receipt_invalid_reason(
        {
            "checkpoint_recovery_execute_handoff_ready": True,
            "checkpoint_recovery_execute_handoff_token_present": True,
            "checkpoint_recovery_execute_handoff_token_sha256": hostile,
            "checkpoint_recovery_execute_handoff_recovery_step_approval_boundary_token_present": True,
            "checkpoint_recovery_execute_handoff_recovery_step_approval_boundary_token_sha256": hostile,
        }
    )
    if invalid_reason != "invalid_or_stale_checkpoint_recovery_execute_handoff":
        raise SystemExit(f"checkpoint handoff invalid reason should tolerate hostile tokens: {invalid_reason!r}")


def assert_recent_tool_runs_approval_holds_are_separate() -> None:
    with TemporaryDirectory(prefix="jarvis-audit-recent-pending-holds-") as temp:
        runtime = make_temp_runtime(Path(temp))
        audit_tools = make_audit_tools(runtime.store)
        recent_tool_runs = audit_tools[0]
        execution_health_report = audit_tools[6]
        blocked = runtime.handle("run command python3 --version")
        if blocked.verified:
            raise SystemExit("high-risk shell smoke should queue approval before execution")
        pending = recent_tool_runs({"limit": 10})
        if "approval held" not in pending.output:
            raise SystemExit(f"pending approval run should render as approval held: {pending.output}")
        if "Next diagnostic: `pending approvals`" not in pending.output:
            raise SystemExit(f"active approval-held recent runs should point to pending approvals: {pending.output}")
        if pending.metadata.get("next_diagnostic_command") != "pending approvals":
            raise SystemExit(f"active approval-held metadata should point to pending approvals: {pending.metadata}")
        if pending.metadata.get("pending_approval_count", 0) < 1:
            raise SystemExit(f"active approval-held metadata should count pending approvals: {pending.metadata}")
        health = execution_health_report({"limit": 10})
        if health.metadata.get("active_approval_holds") != 1:
            raise SystemExit(f"execution health should identify the active approval hold: {health.metadata}")
        if health.metadata.get("failed_or_blocked_runs") != 0:
            raise SystemExit(f"active approval holds must not become execution failures: {health.metadata}")
        if health.metadata.get("newest_problem_run_id") is not None:
            raise SystemExit(f"active approval holds must not become recovery targets: {health.metadata}")
        if health.metadata.get("recovery_closure_blocks_auto_execution") is not False:
            raise SystemExit(f"active approval holds must not deadlock unrelated execution: {health.metadata}")

    with TemporaryDirectory(prefix="jarvis-audit-recent-holds-") as temp:
        runtime = make_temp_runtime(Path(temp))
        recent_tool_runs = make_audit_tools(runtime.store)[0]
        runtime.store.log_tool_run(
            "smoke",
            "send_telegram",
            "HIGH_RISK",
            False,
            False,
            "explicit approval required before execution",
            metadata={"failure_kind": "approval-gate", "requires_confirmation": True},
        )
        held = recent_tool_runs({"limit": 10})
        if not held.ok:
            raise SystemExit(f"recent_tool_runs should render approval holds: {held}")
        if "approval held" not in held.output or "[HIGH_RISK, failed" in held.output:
            raise SystemExit(f"approval hold should not render as a failed run:\n{held.output}")
        held_counts = held.metadata.get("status_counts", {})
        if held_counts.get("approval_held") != 1 or held_counts.get("failed") != 0:
            raise SystemExit(f"approval hold counts should be separate: {held.metadata}")
        if "Next diagnostic: `execution health report`" not in held.output:
            raise SystemExit(f"historical approval-held recent runs should point to execution health: {held.output}")
        if held.metadata.get("next_diagnostic_command") != "execution health report":
            raise SystemExit(f"historical approval-held recent runs metadata should point to execution health: {held.metadata}")
        if held.metadata.get("pending_approval_count") != 0:
            raise SystemExit(f"historical approval-held run should not invent pending approvals: {held.metadata}")
        held_rows = held.metadata.get("recent_tool_run_rows") or []
        if held_rows[0].get("status") != "approval_held" or held_rows[0].get("failure_kind") != "approval_required":
            raise SystemExit(f"approval hold row summary should be canonical: {held_rows}")

        runtime.store.log_tool_run(
            "smoke",
            "send_telegram",
            "HIGH_RISK",
            False,
            True,
            "transport failed after approved attempt",
            metadata={"failure_kind": "transport_error"},
        )
        mixed = recent_tool_runs({"limit": 10})
        mixed_counts = mixed.metadata.get("status_counts", {})
        if mixed_counts.get("approval_held") != 1 or mixed_counts.get("failed") != 1:
            raise SystemExit(f"real failures should still count as failed beside approval holds: {mixed.metadata}")
        if "Next diagnostic: `execution health report`" not in mixed.output:
            raise SystemExit(f"failed recent runs should point to execution health report: {mixed.output}")
        if mixed.metadata.get("next_diagnostic_command") != "execution health report":
            raise SystemExit(f"failed recent runs metadata should point to execution health report: {mixed.metadata}")
        mixed_rows = mixed.metadata.get("recent_tool_run_rows") or []
        statuses = {row.get("status") for row in mixed_rows}
        if not {"approval_held", "failed"}.issubset(statuses):
            raise SystemExit(f"mixed recent tool rows lost status separation: {mixed_rows}")

        marker = "AUDIT_METADATA_SHOULD_NOT_LEAK /\x55sers/example/private/audit.sqlite"
        structured_row = {
            "id": 901,
            "session_id": "structured-audit",
            "tool_name": "send_telegram",
            "risk": "HIGH_RISK",
            "ok": False,
            "approved": False,
            "approval_id": 44,
            "output": "explicit approval required before execution",
            "metadata": {
                "requires_confirmation": HostileMetadataValue(marker),
                "failure_kind": HostileMetadataValue(marker),
                "failure_stage": "explicit approval required",
                "toolset": "messages",
            },
            "created_at": "2026-07-06T11:00:00Z",
        }
        original_recent_tool_runs = runtime.store.recent_tool_runs
        try:
            runtime.store.recent_tool_runs = lambda limit=10: [structured_row][:limit]  # type: ignore[method-assign]
            structured = recent_tool_runs({"limit": 10})
        finally:
            runtime.store.recent_tool_runs = original_recent_tool_runs  # type: ignore[method-assign]
        if not structured.ok:
            raise SystemExit(f"recent_tool_runs should render structured approval-held metadata: {structured}")
        structured_counts = structured.metadata.get("status_counts", {})
        if structured_counts.get("approval_held") != 1 or structured_counts.get("failed") != 0:
            raise SystemExit(f"structured approval hold counts should be separate: {structured.metadata}")
        structured_rows = structured.metadata.get("recent_tool_run_rows") or []
        if structured_rows[0].get("status") != "approval_held" or structured_rows[0].get("failure_kind") != "approval_required":
            raise SystemExit(f"structured approval hold row should be canonical: {structured_rows}")
        if structured_rows[0].get("toolset") != "messages":
            raise SystemExit(f"structured approval hold should preserve safe toolset metadata: {structured_rows}")
        structured_payload = structured.output + repr(structured.metadata)
        if marker in structured_payload or "audit.sqlite" in structured_payload or "/\x55sers/example/private" in structured_payload:
            raise SystemExit(f"recent_tool_runs leaked hostile structured metadata marker: {structured_payload[:1000]}")

        hostile_format_marker = "AUDIT_FORMATTER_METADATA_SHOULD_NOT_LEAK /\x55sers/example/private/formatter.sqlite"
        hostile_format_row = {
            "id": 902,
            "session_id": "formatter-audit",
            "tool_name": "send_telegram",
            "risk": "HIGH_RISK",
            "ok": False,
            "approved": True,
            "approval_id": 45,
            "output": HostileMetadataValue(hostile_format_marker),
            "metadata": {
                "failure_kind": "transport_error",
                "toolset": HostileMetadataValue(hostile_format_marker),
            },
            "created_at": "2026-07-06T11:01:00Z",
        }
        try:
            runtime.store.recent_tool_runs = lambda limit=10: [hostile_format_row][:limit]  # type: ignore[method-assign]
            hostile_format = recent_tool_runs({"limit": 10})
        finally:
            runtime.store.recent_tool_runs = original_recent_tool_runs  # type: ignore[method-assign]
        if not hostile_format.ok:
            raise SystemExit(f"recent_tool_runs should render hostile formatter fields: {hostile_format}")
        hostile_format_counts = hostile_format.metadata.get("status_counts", {})
        if hostile_format_counts.get("failed") != 1 or hostile_format_counts.get("approval_held") != 0:
            raise SystemExit(f"hostile formatter row should stay a real failed run: {hostile_format.metadata}")
        hostile_format_rows = hostile_format.metadata.get("recent_tool_run_rows") or []
        if hostile_format_rows[0].get("toolset") is not None:
            raise SystemExit(f"hostile formatter toolset should be dropped safely: {hostile_format_rows}")
        hostile_format_payload = hostile_format.output + repr(hostile_format.metadata)
        if (
            hostile_format_marker in hostile_format_payload
            or "formatter.sqlite" in hostile_format_payload
            or "/\x55sers/example/private" in hostile_format_payload
        ):
            raise SystemExit(f"recent_tool_runs leaked hostile formatter marker: {hostile_format_payload[:1000]}")


def assert_audit_receipts_tolerate_hostile_output() -> None:
    with TemporaryDirectory(prefix="jarvis-audit-hostile-output-") as temp:
        runtime = make_temp_runtime(Path(temp))
        tools = make_audit_tools(runtime.store)
        verification_receipt = tools[1]
        execution_recovery_packet = tools[4]
        marker = "AUDIT_RECEIPT_OUTPUT_SHOULD_NOT_LEAK /\x55sers/example/private/receipt.sqlite"
        hostile_row = {
            "id": 903,
            "session_id": "hostile-output-audit",
            "tool_name": "send_telegram",
            "risk": "HIGH_RISK",
            "ok": False,
            "approved": True,
            "approval_id": 46,
            "output": HostileMetadataValue(marker),
            "metadata": {
                "failure_kind": "tool_error",
                "toolset": "messages",
                "planned_arg_keys": ["recipient", "message"],
            },
            "created_at": "2026-07-06T11:02:00Z",
        }
        original_recent_tool_runs = runtime.store.recent_tool_runs
        original_get_tool_run = runtime.store.get_tool_run
        try:
            runtime.store.recent_tool_runs = lambda limit=10: [hostile_row][:limit]  # type: ignore[method-assign]
            runtime.store.get_tool_run = lambda run_id: hostile_row if int(run_id) == 903 else None  # type: ignore[method-assign]
            receipt = verification_receipt({"run_id": "latest"})
            recovery = execution_recovery_packet({"run_id": 903})
        finally:
            runtime.store.recent_tool_runs = original_recent_tool_runs  # type: ignore[method-assign]
            runtime.store.get_tool_run = original_get_tool_run  # type: ignore[method-assign]

        if not receipt.ok:
            raise SystemExit(f"verification_receipt should tolerate hostile output: {receipt}")
        receipt_handoff = receipt.metadata.get("verification_receipt_handoff") or {}
        if (
            receipt.metadata.get("verdict") != "APPROVAL_EVIDENCE_MISSING"
            or receipt.metadata.get("approved") is not False
            or receipt.metadata.get("output_chars") != 0
            or receipt_handoff.get("approval_problem") is not True
        ):
            raise SystemExit(f"verification_receipt should fail closed on forged risky failure evidence: {receipt.metadata}")
        if "(empty output)" not in receipt.output:
            raise SystemExit(f"verification_receipt should render hostile output as empty evidence: {receipt.output[:1000]}")

        if not recovery.ok:
            raise SystemExit(f"execution_recovery_packet should tolerate hostile output: {recovery}")
        if recovery.metadata.get("verdict") != "APPROVAL_EVIDENCE_MISSING":
            raise SystemExit(f"execution_recovery_packet should prioritize forged approval evidence: {recovery.metadata}")
        if recovery.metadata.get("output_approval_ids") != []:
            raise SystemExit(f"execution_recovery_packet should not invent approval ids from hostile output: {recovery.metadata}")

        payload = receipt.output + repr(receipt.metadata) + recovery.output + repr(recovery.metadata)
        if marker in payload or "receipt.sqlite" in payload or "/\x55sers/example/private" in payload:
            raise SystemExit(f"audit receipt/recovery leaked hostile output marker: {payload[:1000]}")


def main() -> None:
    assert_audit_metadata_bool_is_exact()
    assert_audit_formatters_tolerate_hostile_values()
    assert_runtime_result_receipt_bools_are_exact()
    assert_recent_tool_runs_approval_holds_are_separate()
    assert_audit_receipts_tolerate_hostile_output()
    with TemporaryDirectory(prefix="jarvis-audit-") as temp:
        runtime = make_temp_runtime(Path(temp))
        stage_metadata = runtime._tool_run_audit_metadata(
            ToolResult(
                "synthetic_channel_failure",
                False,
                "synthetic channel failure",
                {
                    "telegram_call_stage": "telegram_exact_chat_not_found",
                    "instagram_send_stage": "instagram_new_recipient_not_found",
                    "instagram_call_stage": "instagram_call_button_not_found",
                    "kakao_send_stage": "kakao_chat_not_found",
                    "kakao_call_stage": "kakao_call_button_not_found",
                    "imessage_send_stage": "contact_resolution_not_found",
                    "call_stage": "facetime_contact_not_found",
                    "failure_stage": "channel_transport_failed",
                },
            )
        )
        for key, expected in {
            "telegram_call_stage": "telegram_exact_chat_not_found",
            "instagram_send_stage": "instagram_new_recipient_not_found",
            "instagram_call_stage": "instagram_call_button_not_found",
            "kakao_send_stage": "kakao_chat_not_found",
            "kakao_call_stage": "kakao_call_button_not_found",
            "imessage_send_stage": "contact_resolution_not_found",
            "call_stage": "facetime_contact_not_found",
            "failure_stage": "channel_transport_failed",
        }.items():
            if stage_metadata.get(key) != expected:
                raise SystemExit(f"runtime audit metadata dropped {key}: {stage_metadata}")
        hostile_stage = runtime._tool_run_audit_metadata(
            ToolResult(
                "synthetic_channel_failure",
                False,
                "synthetic hostile channel failure",
                {"instagram_send_stage": HostileMetadataValue("SHOULD_NOT_SERIALIZE")},
            )
        )
        if "instagram_send_stage" in hostile_stage:
            raise SystemExit(f"runtime audit metadata retained a non-string channel stage: {hostile_stage}")
        assert_runtime_result_trace_bools_are_exact(runtime)
        assert_runtime_trace_handler_execution_truth(runtime)
        recent_tool_runs, verification_receipt, runtime_trace_receipt, execution_audit_gate, execution_recovery_packet, after_action_learning_packet, execution_health_report, recovery_closure_checklist, execution_learning_closure_packet = make_audit_tools(runtime.store)
        reversed_latest_proof_routes = {
            "verification please": ("verification_receipt", {"run_id": "latest", "expectation": ""}),
            "verification receipts please": ("verification_receipt", {"run_id": "latest", "expectation": ""}),
            "verification receipt latest please": ("verification_receipt", {"run_id": "latest", "expectation": ""}),
            "show latest verification": ("verification_receipt", {"run_id": "latest", "expectation": ""}),
            "show latest verification receipt": ("verification_receipt", {"run_id": "latest", "expectation": ""}),
            "show current verification receipt": ("verification_receipt", {"run_id": "latest", "expectation": ""}),
            "show newest verification receipt": ("verification_receipt", {"run_id": "latest", "expectation": ""}),
            "show last verification receipt": ("verification_receipt", {"run_id": "latest", "expectation": ""}),
            "verification receipt for run 1": ("verification_receipt", {"run_id": "1", "expectation": ""}),
            "show receipt for run 1": ("verification_receipt", {"run_id": "1", "expectation": ""}),
            "verify receipt for run 1: output includes Python version": (
                "verification_receipt",
                {"run_id": "1", "expectation": "output includes Python version"},
            ),
            "did run 1 pass verification": ("verification_receipt", {"run_id": "1", "expectation": ""}),
            "was run 1 verified": ("verification_receipt", {"run_id": "1", "expectation": ""}),
            "is run 1 verified": ("verification_receipt", {"run_id": "1", "expectation": ""}),
            "verify run 1": ("verification_receipt", {"run_id": "1", "expectation": ""}),
            "recovery please": ("execution_recovery_packet", {}),
            "execution recovery latest please": ("execution_recovery_packet", {}),
            "show latest recovery": ("execution_recovery_packet", {}),
            "show latest execution recovery": ("execution_recovery_packet", {}),
            "show current execution recovery": ("execution_recovery_packet", {}),
            "show newest recovery packet": ("execution_recovery_packet", {}),
            "show last recovery packet": ("execution_recovery_packet", {}),
            "show recovery for run 1": ("execution_recovery_packet", {"run_id": "1"}),
            "show execution recovery for run 1": ("execution_recovery_packet", {"run_id": "1"}),
            "how do we recover run 1": ("execution_recovery_packet", {"run_id": "1"}),
            "how should jarvis recover run 1": ("execution_recovery_packet", {"run_id": "1"}),
            "what should jarvis do after run 1 failed": ("execution_recovery_packet", {"run_id": "1"}),
            "what should jarvis do after failed run 1": ("execution_recovery_packet", {"run_id": "1"}),
            "what failed in run 1": ("execution_recovery_packet", {"run_id": "1"}),
            "why did run 1 fail": ("execution_recovery_packet", {"run_id": "1"}),
            "after-action learning latest please": ("after_action_learning_packet", {}),
            "show latest after-action learning packet": ("after_action_learning_packet", {}),
            "show current after-action learning packet": ("after_action_learning_packet", {}),
            "show newest after-action learning": ("after_action_learning_packet", {}),
            "show last after action learning": ("after_action_learning_packet", {}),
            "show learning for run 1": ("after_action_learning_packet", {"run_id": "1"}),
            "show after-action learning for run 1": ("after_action_learning_packet", {"run_id": "1"}),
            "execution learning closure latest please": ("execution_learning_closure_packet", {}),
            "show latest execution learning closure": ("execution_learning_closure_packet", {}),
            "show current learning closure": ("execution_learning_closure_packet", {}),
            "show newest learning closure packet": ("execution_learning_closure_packet", {}),
            "show last execution learning closure packet": ("execution_learning_closure_packet", {}),
            "close learning for run 1": ("execution_learning_closure_packet", {"run_id": "1"}),
            "show learning closure for run 1": ("execution_learning_closure_packet", {"run_id": "1"}),
        }
        for command, (expected_tool, expected_args) in reversed_latest_proof_routes.items():
            plan = runtime.planner.plan(command)
            if len(plan.actions) != 1:
                raise SystemExit(f"Expected exactly one action for reversed proof route {command!r}: {plan.actions}")
            action = plan.actions[0]
            if action.tool_name != expected_tool or action.args != expected_args:
                raise SystemExit(f"Reversed proof route misplanned {command!r}: {(action.tool_name, action.args)}")
        natural_run_proof_routes = {
            "receipt for run 1": ("verification_receipt", {"run_id": "1", "expectation": ""}),
            "verification for run 1": ("verification_receipt", {"run_id": "1", "expectation": ""}),
            "prove run 1": ("verification_receipt", {"run_id": "1", "expectation": ""}),
            "proof for run 1": ("verification_receipt", {"run_id": "1", "expectation": ""}),
            "what happened in run 1": ("verification_receipt", {"run_id": "1", "expectation": ""}),
            "what happened with run 1": ("verification_receipt", {"run_id": "1", "expectation": ""}),
            "trace run 1": ("verification_receipt", {"run_id": "1", "expectation": ""}),
            "runtime trace for message 1": ("runtime_trace_receipt", {"message_id": "1"}),
            "trace for message 1": ("runtime_trace_receipt", {"message_id": "1"}),
            "audit run 1": ("execution_audit_gate", {}),
            "audit for run 1": ("execution_audit_gate", {}),
            "execution audit for run 1": ("execution_audit_gate", {}),
            "run integrity for run 1": ("execution_audit_gate", {}),
            "recovery for run 1": ("execution_recovery_packet", {"run_id": "1"}),
            "learning for run 1": ("after_action_learning_packet", {"run_id": "1"}),
            "after action for run 1": ("after_action_learning_packet", {"run_id": "1"}),
            "after action learning for run 1": ("after_action_learning_packet", {"run_id": "1"}),
            "what did we learn from run 1": ("after_action_learning_packet", {"run_id": "1"}),
            "is run 1 learning closed": ("execution_learning_closure_packet", {"run_id": "1"}),
            "is learning closed for run 1": ("execution_learning_closure_packet", {"run_id": "1"}),
            "learning closure for run 1": ("execution_learning_closure_packet", {"run_id": "1"}),
            "execution learning closure for run 1": ("execution_learning_closure_packet", {"run_id": "1"}),
        }
        for command, (expected_tool, expected_args) in natural_run_proof_routes.items():
            plan = runtime.planner.plan(command)
            if len(plan.actions) != 1:
                raise SystemExit(f"Expected exactly one action for natural run proof route {command!r}: {plan.actions}")
            action = plan.actions[0]
            if action.tool_name != expected_tool or action.args != expected_args:
                raise SystemExit(f"Natural run proof route misplanned {command!r}: {(action.tool_name, action.args)}")
        polite_diagnostic_routes = {
            "runtime trace receipt please": ("runtime_trace_receipt", {}),
            "show runtime trace receipt": ("runtime_trace_receipt", {}),
            "show latest runtime trace receipt": ("runtime_trace_receipt", {}),
            "show current runtime trace receipt": ("runtime_trace_receipt", {}),
            "show newest runtime trace": ("runtime_trace_receipt", {}),
            "show last trace receipt": ("runtime_trace_receipt", {}),
            "execution audit please": ("execution_audit_gate", {}),
            "show execution audit": ("execution_audit_gate", {}),
            "show latest execution audit": ("execution_audit_gate", {}),
            "show current execution audit gate": ("execution_audit_gate", {}),
            "show newest audit gate": ("execution_audit_gate", {}),
            "show last run integrity report": ("execution_audit_gate", {}),
            "show latest execution health report": ("execution_health_report", {}),
            "show current execution health": ("execution_health_report", {}),
            "show newest runtime health": ("execution_health_report", {}),
            "show last tool health": ("execution_health_report", {}),
            "tool health please": ("execution_health_report", {}),
            "what runs failed recently": ("execution_health_report", {}),
            "show failed runs": ("execution_health_report", {}),
            "show blocked runs": ("execution_health_report", {}),
            "recent failed runs": ("execution_health_report", {}),
            "recent blocked runs": ("execution_health_report", {}),
            "why is execution broken": ("execution_health_report", {}),
            "what broke in execution": ("execution_health_report", {}),
            "recent tool failures": ("execution_health_report", {}),
            "recent failed tool runs": ("execution_health_report", {}),
            "recent tool runs failed": ("execution_health_report", {}),
            "show latest recovery closure checklist": ("recovery_closure_checklist", {}),
            "show current recovery closure": ("recovery_closure_checklist", {}),
            "show newest execution closure checklist": ("recovery_closure_checklist", {}),
            "show last recovery closure": ("recovery_closure_checklist", {}),
        }
        for command, (expected_tool, expected_args) in polite_diagnostic_routes.items():
            plan = runtime.planner.plan(command)
            if len(plan.actions) != 1:
                raise SystemExit(f"Expected exactly one action for diagnostic route {command!r}: {plan.actions}")
            action = plan.actions[0]
            if action.tool_name != expected_tool or action.args != expected_args:
                raise SystemExit(f"Diagnostic route misplanned {command!r}: {(action.tool_name, action.args)}")
        tool_run_diagnostic_routes = {
            "audit please": ("recent_tool_runs", {}),
            "recent tool runs please": ("recent_tool_runs", {}),
            "tool runs please": ("recent_tool_runs", {}),
            "audit log please": ("recent_tool_runs", {}),
            "show latest audit": ("recent_tool_runs", {}),
            "recent audit please": ("recent_tool_runs", {}),
            "what did you run please": ("recent_tool_runs", {}),
            "show recent tool runs": ("recent_tool_runs", {}),
            "show latest tool runs": ("recent_tool_runs", {}),
            "show current tool runs": ("recent_tool_runs", {}),
            "show newest audit log": ("recent_tool_runs", {}),
            "show last audit log": ("recent_tool_runs", {}),
            "show me recent audit": ("recent_tool_runs", {}),
            "show me what you ran": ("recent_tool_runs", {}),
            "show latest recent tool runs": ("recent_tool_runs", {}),
        }
        for command, (expected_tool, expected_args) in tool_run_diagnostic_routes.items():
            plan = runtime.planner.plan(command)
            if len(plan.actions) != 1:
                raise SystemExit(f"Expected exactly one action for recent tool run route {command!r}: {plan.actions}")
            action = plan.actions[0]
            if action.tool_name != expected_tool or action.args != expected_args:
                raise SystemExit(f"Recent tool run route misplanned {command!r}: {(action.tool_name, action.args)}")
        cases = [
            ("calculate 2 + 2", False),
            ("run command python3 --version", False),
            ("approval readiness 1", False),
            ("approval packet 1", False),
            ("approve approval 1", False),
            ("recent tool runs", False),
            ("verification receipt 6: command output includes Python version", False),
            ("runtime trace receipt", False),
            ("execution recovery", False),
            ("what should Jarvis do after the failed run", False),
            ("after-action learning", False),
            ("what can Jarvis learn from run 2", False),
            ("execution health report", False),
            ("is execution healthy", False),
            ("recovery closure checklist", False),
            ("execution learning closure", False),
        ]
        for case, approved in cases:
            result = handle_runtime_case(runtime, case, approved=approved)
            status = "ok" if result.verified else "blocked"
            approval = " approved" if approved else ""
            print(f"[{status}{approval}] {case}")
            print(result.response[:1400])
            print()
            if case == "recent tool runs":
                for expected in [
                    "Recent tool runs",
                    "calculate",
                    "run_shell_command",
                    "approved",
                    "approval #",
                    "toolset code",
                ]:
                    if expected not in result.response:
                        raise SystemExit(f"recent tool runs missing expected text: {expected}")
                metadata = result.tool_results[0].metadata
                if metadata.get("approved_linked_runs", 0) < 1:
                    raise SystemExit("recent tool runs metadata missed linked approved run count.")
                if metadata.get("audit_toolsets", {}).get("code", 0) < 1:
                    raise SystemExit(f"recent tool runs metadata missed audit toolset counts: {metadata}")
                handoff = metadata.get("recent_tool_runs_handoff") or {}
                if handoff.get("source") != "recent_tool_runs":
                    raise SystemExit(f"recent tool runs missed structured handoff: {metadata}")
                for nested_key, flat_key in {
                    "count": "count",
                    "limit": "limit",
                    "approved_linked_runs": "approved_linked_runs",
                    "audit_toolsets": "audit_toolsets",
                    "risk_counts": "risk_counts",
                    "status_counts": "status_counts",
                    "rows": "recent_tool_run_rows",
                    "readable_tool_run_rows": "readable_tool_run_rows",
                    "unreadable_tool_run_rows": "unreadable_tool_run_rows",
                }.items():
                    if handoff.get(nested_key) != metadata.get(flat_key):
                        raise SystemExit(f"recent tool runs handoff field {nested_key} diverged from {flat_key}: {metadata}")
                rows = handoff.get("rows") or []
                if len(rows) != metadata.get("count") or not any(row.get("tool_name") == "run_shell_command" and row.get("toolset") == "code" for row in rows):
                    raise SystemExit(f"recent tool runs handoff missed row summaries: {metadata}")
                if handoff.get("risk_counts", {}).get("HIGH_RISK", 0) < 1 or handoff.get("status_counts", {}).get("approval_held", 0) < 1:
                    raise SystemExit(f"recent tool runs handoff missed risk/status summaries: {metadata}")
                for key in ["review_only", "draft_only", "loads_without_execution"]:
                    if handoff.get(key) is not True or metadata.get(f"recent_tool_runs_{key}") is not True:
                        raise SystemExit(f"recent tool runs handoff missed true {key}: {metadata}")
                for key in [
                    "authorizes_execution",
                    "authorizes_completion_claim",
                    "approval_granted",
                    "calls_model",
                    "executes_tools",
                    "writes_files",
                    "reads_personal_data",
                    "external_side_effect",
                    "controls_computer",
                    "queues_approval",
                ]:
                    if handoff.get(key) is not False:
                        raise SystemExit(f"recent tool runs handoff should keep {key}=False: {metadata}")
                if (
                    metadata.get("recent_tool_runs_authorizes_execution") is not False
                    or metadata.get("recent_tool_runs_authorizes_completion_claim") is not False
                    or metadata.get("recent_tool_runs_approval_granted") is not False
                ):
                    raise SystemExit(f"recent tool runs flat authority fields should be false: {metadata}")
                assert_no_future_authority(metadata, "recent tool runs")
                if (
                    metadata.get("executes_tools")
                    or metadata.get("queues_approval")
                    or metadata.get("approves_request")
                    or metadata.get("dismisses_request")
                    or metadata.get("reads_private_data")
                    or metadata.get("writes_files")
                    or metadata.get("controls_computer")
                ):
                    raise SystemExit("recent tool runs should remain read-only.")
                marker = "SHOULD_NOT_LEAK_AUDIT_HOSTILE_ROW"
                readable_rows = runtime.store.recent_tool_runs(5)
                original_recent_tool_runs = runtime.store.recent_tool_runs

                def hostile_recent_tool_runs(_limit: int):
                    return [HostileRow(marker), *readable_rows]

                try:
                    runtime.store.recent_tool_runs = hostile_recent_tool_runs  # type: ignore[method-assign]
                    hostile_result = recent_tool_runs({"limit": 5})
                finally:
                    runtime.store.recent_tool_runs = original_recent_tool_runs  # type: ignore[method-assign]
                if not hostile_result.ok:
                    raise SystemExit(f"recent_tool_runs should tolerate malformed audit rows: {hostile_result}")
                hostile_metadata = hostile_result.metadata
                if hostile_metadata.get("count") != len(readable_rows):
                    raise SystemExit(f"recent_tool_runs should count readable rows only: {hostile_metadata}")
                if hostile_metadata.get("readable_tool_run_rows") != len(readable_rows):
                    raise SystemExit(f"recent_tool_runs missed readable row count: {hostile_metadata}")
                if hostile_metadata.get("unreadable_tool_run_rows") != 1:
                    raise SystemExit(f"recent_tool_runs missed unreadable row count: {hostile_metadata}")
                if "unreadable tool run row(s) hidden for safety" not in hostile_result.output:
                    raise SystemExit(f"recent_tool_runs should report hidden unreadable rows: {hostile_result.output}")
                hostile_handoff = hostile_metadata.get("recent_tool_runs_handoff") or {}
                if hostile_handoff.get("unreadable_tool_run_rows") != 1:
                    raise SystemExit(f"recent_tool_runs handoff missed unreadable row count: {hostile_metadata}")
                if len(hostile_handoff.get("rows") or []) != len(readable_rows):
                    raise SystemExit(f"recent_tool_runs handoff should preserve readable rows: {hostile_metadata}")
                hostile_payload = hostile_result.output + json.dumps(hostile_metadata, sort_keys=True, default=str)
                if marker in hostile_payload:
                    raise SystemExit("recent_tool_runs leaked raw malformed audit row text.")
                all_hostile_marker = "SHOULD_NOT_LEAK_ALL_HOSTILE_AUDIT_ROW"

                def all_hostile_recent_tool_runs(_limit: int):
                    return [HostileRow(all_hostile_marker)]

                try:
                    runtime.store.recent_tool_runs = all_hostile_recent_tool_runs  # type: ignore[method-assign]
                    all_hostile_result = recent_tool_runs({"limit": 5})
                finally:
                    runtime.store.recent_tool_runs = original_recent_tool_runs  # type: ignore[method-assign]
                if not all_hostile_result.ok:
                    raise SystemExit(f"recent_tool_runs should tolerate all malformed audit rows: {all_hostile_result}")
                all_hostile_metadata = all_hostile_result.metadata
                if all_hostile_metadata.get("count") != 0 or all_hostile_metadata.get("readable_tool_run_rows") != 0:
                    raise SystemExit(f"recent_tool_runs should count no readable rows for all-hostile input: {all_hostile_metadata}")
                if all_hostile_metadata.get("unreadable_tool_run_rows") != 1:
                    raise SystemExit(f"recent_tool_runs missed all-hostile unreadable row count: {all_hostile_metadata}")
                if "Next diagnostic: `jarvis doctor`" not in all_hostile_result.output:
                    raise SystemExit(f"recent_tool_runs should provide an all-hostile recovery hint: {all_hostile_result.output}")
                if all_hostile_metadata.get("next_diagnostic_command") != "jarvis doctor":
                    raise SystemExit(f"recent_tool_runs missed all-hostile next diagnostic metadata: {all_hostile_metadata}")
                all_hostile_handoff = all_hostile_metadata.get("recent_tool_runs_handoff") or {}
                if all_hostile_handoff.get("next_diagnostic_command") != "jarvis doctor":
                    raise SystemExit(f"recent_tool_runs handoff missed all-hostile next diagnostic: {all_hostile_metadata}")
                all_hostile_payload = all_hostile_result.output + json.dumps(all_hostile_metadata, sort_keys=True, default=str)
                if all_hostile_marker in all_hostile_payload:
                    raise SystemExit("recent_tool_runs leaked all-hostile malformed audit row text.")
            if case.startswith("verification receipt"):
                for expected in [
                    "Jarvis verification receipt",
                    "after-action proof layer",
                    "Evidence:",
                    "run_shell_command",
                    "approved: yes",
                    "expected outcome",
                    "Verdict:",
                    "OUTPUT_REVIEW_REQUIRED",
                    "does not call models",
                ]:
                    if expected not in result.response:
                        raise SystemExit(f"verification receipt missing expected text: {expected}")
                metadata = result.tool_results[0].metadata
                if metadata.get("tool_name") != "run_shell_command" or metadata.get("approved") is not True:
                    raise SystemExit(f"verification receipt missed approved run evidence: {metadata}")
                if metadata.get("verdict") != "OUTPUT_REVIEW_REQUIRED":
                    raise SystemExit(f"verification receipt missed expectation review verdict: {metadata}")
                receipt_handoff = metadata.get("verification_receipt_handoff") or {}
                if receipt_handoff.get("source") != "verification_receipt":
                    raise SystemExit(f"verification receipt missed structured handoff: {metadata}")
                for nested_key, flat_key in {
                    "found": "found",
                    "verdict": "verdict",
                    "expectation": "expectation",
                    "approval_evidence_required": "approval_evidence_required",
                    "output_chars": "output_chars",
                }.items():
                    if receipt_handoff.get(nested_key) != metadata.get(flat_key):
                        raise SystemExit(f"verification receipt handoff field {nested_key} diverged from {flat_key}: {metadata}")
                target = receipt_handoff.get("target") or {}
                for nested_key, flat_key in {
                    "run_id": "run_id",
                    "tool_name": "tool_name",
                    "risk": "risk",
                    "ok": "ok",
                    "approved": "approved",
                    "approval_id": "approval_id",
                    "toolset": "toolset",
                    "failure_kind": "failure_kind",
                    "planned_arg_keys": "planned_arg_keys",
                }.items():
                    if target.get(nested_key) != metadata.get(flat_key):
                        raise SystemExit(f"verification receipt target handoff field {nested_key} diverged: {metadata}")
                for key in ["review_only", "draft_only", "loads_without_execution"]:
                    if receipt_handoff.get(key) is not True or metadata.get(f"verification_receipt_{key}") is not True:
                        raise SystemExit(f"verification receipt handoff missed true {key}: {metadata}")
                for key in [
                    "authorizes_execution",
                    "authorizes_completion_claim",
                    "approval_granted",
                    "calls_model",
                    "executes_tools",
                    "writes_files",
                    "reads_personal_data",
                    "external_side_effect",
                    "controls_computer",
                    "queues_approval",
                ]:
                    if receipt_handoff.get(key) is not False:
                        raise SystemExit(f"verification receipt handoff should keep {key}=False: {metadata}")
                if (
                    metadata.get("verification_receipt_authorizes_execution") is not False
                    or metadata.get("verification_receipt_authorizes_completion_claim") is not False
                    or metadata.get("verification_receipt_approval_granted") is not False
                ):
                    raise SystemExit(f"verification receipt flat authority fields should be false: {metadata}")
                assert_no_future_authority(metadata, "verification receipt")
                blocked_receipt = verification_receipt({"run_id": 2})
                if not blocked_receipt.ok or blocked_receipt.metadata.get("verdict") != "FAILED_OR_BLOCKED":
                    raise SystemExit(f"blocked verification receipt should inspect held run: {blocked_receipt}")
                assert_no_future_authority(blocked_receipt.metadata, "blocked verification receipt")
                if blocked_receipt.metadata.get("toolset") != "code" or blocked_receipt.metadata.get("failure_kind") != "approval_required":
                    raise SystemExit(f"blocked verification receipt missed stored audit metadata: {blocked_receipt.metadata}")
                if blocked_receipt.metadata.get("planned_arg_keys") != ["command"]:
                    raise SystemExit(f"blocked verification receipt missed planned arg keys: {blocked_receipt.metadata}")
                blocked_handoff = blocked_receipt.metadata.get("verification_receipt_handoff") or {}
                if blocked_handoff.get("target", {}).get("run_id") != 2 or blocked_handoff.get("verdict") != "FAILED_OR_BLOCKED":
                    raise SystemExit(f"blocked verification receipt missed handoff target/verdict: {blocked_receipt.metadata}")
                if blocked_handoff.get("approval_problem") is not False:
                    raise SystemExit(f"approval-held run should not be treated as invoked approval evidence: {blocked_receipt.metadata}")
                if (
                    metadata.get("executes_tools")
                    or metadata.get("queues_approval")
                    or metadata.get("approves_request")
                    or metadata.get("dismisses_request")
                    or metadata.get("reads_private_data")
                    or metadata.get("writes_files")
                    or metadata.get("controls_computer")
                ):
                    raise SystemExit("verification receipt should remain read-only.")
                marker = "SHOULD_NOT_LEAK_VERIFICATION_RECEIPT_HOSTILE_ROW"
                readable_rows = runtime.store.recent_tool_runs(5)
                original_recent_tool_runs = runtime.store.recent_tool_runs

                def hostile_recent_for_receipt(*_args, **_kwargs):
                    return [HostileRow(marker), *readable_rows]

                try:
                    runtime.store.recent_tool_runs = hostile_recent_for_receipt  # type: ignore[method-assign]
                    hostile_latest = verification_receipt({"run_id": "latest"})
                finally:
                    runtime.store.recent_tool_runs = original_recent_tool_runs  # type: ignore[method-assign]
                if not hostile_latest.ok:
                    raise SystemExit(f"verification_receipt latest should skip malformed recent rows: {hostile_latest}")
                hostile_latest_metadata = hostile_latest.metadata
                if hostile_latest_metadata.get("run_id") != int(readable_rows[0]["id"]):
                    raise SystemExit(f"verification_receipt latest chose wrong readable row: {hostile_latest_metadata}")
                if hostile_latest_metadata.get("readable_tool_run_rows") != len(readable_rows):
                    raise SystemExit(f"verification_receipt latest missed readable row count: {hostile_latest_metadata}")
                if hostile_latest_metadata.get("unreadable_tool_run_rows") != 1:
                    raise SystemExit(f"verification_receipt latest missed unreadable row count: {hostile_latest_metadata}")
                hostile_latest_handoff = hostile_latest_metadata.get("verification_receipt_handoff") or {}
                if hostile_latest_handoff.get("unreadable_tool_run_rows") != 1:
                    raise SystemExit(f"verification_receipt handoff missed unreadable rows: {hostile_latest_metadata}")
                if "unreadable tool run row(s) hidden for safety" not in hostile_latest.output:
                    raise SystemExit(f"verification_receipt latest should report hidden unreadable rows: {hostile_latest.output}")
                hostile_latest_payload = hostile_latest.output + json.dumps(hostile_latest_metadata, sort_keys=True, default=str)
                if marker in hostile_latest_payload:
                    raise SystemExit("verification_receipt latest leaked raw malformed audit row text.")

                original_get_tool_run = runtime.store.get_tool_run

                def hostile_get_tool_run(_run_id: int):
                    return HostileRow(marker)

                try:
                    runtime.store.get_tool_run = hostile_get_tool_run  # type: ignore[method-assign]
                    hostile_exact = verification_receipt({"run_id": int(readable_rows[0]["id"])})
                finally:
                    runtime.store.get_tool_run = original_get_tool_run  # type: ignore[method-assign]
                if hostile_exact.ok or hostile_exact.metadata.get("verdict") != "UNREADABLE_RUN":
                    raise SystemExit(f"verification_receipt exact unreadable row should fail closed: {hostile_exact}")
                if hostile_exact.metadata.get("unreadable_tool_run_rows", 0) < 1:
                    raise SystemExit(f"verification_receipt exact missed unreadable row metadata: {hostile_exact.metadata}")
                assert_audit_lookup_recovery(
                    hostile_exact,
                    "verification receipt unreadable run",
                    ["setup check", "recent tool runs", "verification receipt latest"],
                )
                if hostile_exact.metadata.get("retry_requires_storage_repair") is not True:
                    raise SystemExit(f"Unreadable audit row should require storage repair: {hostile_exact.metadata}")
                hostile_exact_payload = hostile_exact.output + json.dumps(hostile_exact.metadata, sort_keys=True, default=str)
                if marker in hostile_exact_payload:
                    raise SystemExit("verification_receipt exact leaked raw malformed audit row text.")
            if case == "runtime trace receipt":
                for expected in [
                    "Jarvis runtime trace receipt",
                    "stage-by-stage receipt",
                    "Trace identity",
                    "Lifecycle stages",
                    "planning",
                    "Planner diagnostics",
                    "model planner: not attempted",
                    "permission",
                    "execution",
                    "verification",
                    "Planned actions",
                    "Tool result evidence",
                    "Safety and approval state",
                    "approval queue before: 0",
                    "approval queue after: 0",
                    "approval queue delta: 0",
                    "queued approval ids: none",
                    "new approval ids: none",
                    "reused approval ids: none",
                    "referenced approval ids: 1",
                    "does not call models",
                ]:
                    if expected not in result.response:
                        raise SystemExit(f"runtime trace receipt missing expected text: {expected}")
                metadata = result.tool_results[0].metadata
                if not metadata.get("found") or metadata.get("stages", 0) < 1:
                    raise SystemExit(f"runtime trace receipt missed trace metadata: {metadata}")
                if metadata.get("route") != "tools":
                    raise SystemExit(f"runtime trace receipt should inspect a tool route: {metadata}")
                if metadata.get("queued_approvals") != 0 or metadata.get("referenced_approval_ids") != [1]:
                    raise SystemExit(f"runtime trace receipt conflated queued and referenced approval ids: {metadata}")
                if (
                    metadata.get("approval_queue_before") != 0
                    or metadata.get("approval_queue_after") != 0
                    or metadata.get("approval_queue_delta") != 0
                    or metadata.get("new_approval_ids") != []
                    or metadata.get("reused_approval_ids") != []
                ):
                    raise SystemExit(f"runtime trace receipt missed approval queue ledger: {metadata}")
                trace_handoff = metadata.get("runtime_trace_receipt_handoff") or {}
                if trace_handoff.get("source") != "runtime_trace_receipt":
                    raise SystemExit(f"runtime trace receipt missed structured handoff: {metadata}")
                for nested_key, flat_key in {
                    "found": "found",
                    "message_id": "message_id",
                    "session_id": "session_id",
                    "route": "route",
                    "verdict": "verdict",
                    "verified": "verified",
                    "approval_required": "approval_required",
                    "risk_levels": "risk_levels",
                    "result_toolsets": "result_toolsets",
                    "planner_notes": "planner_notes",
                    "planner_metadata": "planner_metadata",
                }.items():
                    if trace_handoff.get(nested_key) != metadata.get(flat_key):
                        raise SystemExit(f"runtime trace handoff field {nested_key} diverged from {flat_key}: {metadata}")
                if metadata.get("planner_model_planner_attempted") is not False:
                    raise SystemExit(f"runtime trace should report no model-planner attempt for deterministic traces: {metadata}")
                counts = trace_handoff.get("counts") or {}
                for nested_key, flat_key in {
                    "stages": "stages",
                    "planned_actions": "planned_actions",
                    "tool_results": "tool_results",
                    "inspected_messages": "inspected_messages",
                    "readable_message_rows": "readable_message_rows",
                    "unreadable_message_rows": "unreadable_message_rows",
                }.items():
                    if counts.get(nested_key) != metadata.get(flat_key):
                        raise SystemExit(f"runtime trace handoff count {nested_key} diverged from {flat_key}: {metadata}")
                approval = trace_handoff.get("approval") or {}
                for nested_key, flat_key in {
                    "queued_approvals": "queued_approvals",
                    "new_approval_count": "new_approval_count",
                    "reused_approval_count": "reused_approval_count",
                    "new_approval_ids": "new_approval_ids",
                    "reused_approval_ids": "reused_approval_ids",
                    "approval_queue_before": "approval_queue_before",
                    "approval_queue_after": "approval_queue_after",
                    "approval_queue_delta": "approval_queue_delta",
                    "referenced_approvals": "referenced_approvals",
                    "referenced_approval_ids": "referenced_approval_ids",
                    "approved_reruns": "approved_reruns",
                    "approved_rerun_run_ids": "approved_rerun_run_ids",
                    "approved_rerun_approval_ids": "approved_rerun_approval_ids",
                }.items():
                    if approval.get(nested_key) != metadata.get(flat_key):
                        raise SystemExit(f"runtime trace handoff approval field {nested_key} diverged from {flat_key}: {metadata}")
                for key in ["review_only", "draft_only", "loads_without_execution"]:
                    if trace_handoff.get(key) is not True or metadata.get(f"runtime_trace_receipt_{key}") is not True:
                        raise SystemExit(f"runtime trace handoff missed true {key}: {metadata}")
                for key in [
                    "authorizes_execution",
                    "authorizes_completion_claim",
                    "approval_granted",
                    "calls_model",
                    "executes_tools",
                    "writes_files",
                    "reads_personal_data",
                    "external_side_effect",
                    "controls_computer",
                    "queues_approval",
                ]:
                    if trace_handoff.get(key) is not False:
                        raise SystemExit(f"runtime trace handoff should keep {key}=False: {metadata}")
                if (
                    metadata.get("runtime_trace_receipt_authorizes_execution") is not False
                    or metadata.get("runtime_trace_receipt_authorizes_completion_claim") is not False
                    or metadata.get("runtime_trace_receipt_approval_granted") is not False
                ):
                    raise SystemExit(f"runtime trace receipt flat authority fields should be false: {metadata}")
                assert_no_future_authority(metadata, "runtime trace receipt")
                if (
                    metadata.get("executes_tools")
                    or metadata.get("queues_approval")
                    or metadata.get("approves_request")
                    or metadata.get("dismisses_request")
                    or metadata.get("reads_private_data")
                    or metadata.get("writes_files")
                    or metadata.get("controls_computer")
                ):
                    raise SystemExit("runtime trace receipt should remain read-only.")
                marker = "SHOULD_NOT_LEAK_RUNTIME_TRACE_HOSTILE_MESSAGE_ROW"
                baseline_latest_trace = runtime_trace_receipt({})
                if not baseline_latest_trace.ok:
                    raise SystemExit(f"runtime_trace_receipt latest baseline failed: {baseline_latest_trace}")
                baseline_latest_metadata = baseline_latest_trace.metadata
                readable_messages = runtime.store.recent_messages(limit=40)
                original_recent_messages = runtime.store.recent_messages

                def hostile_recent_messages(*_args, **_kwargs):
                    return [HostileRow(marker), *readable_messages]

                try:
                    runtime.store.recent_messages = hostile_recent_messages  # type: ignore[method-assign]
                    hostile_trace = runtime_trace_receipt({})
                finally:
                    runtime.store.recent_messages = original_recent_messages  # type: ignore[method-assign]
                if not hostile_trace.ok:
                    raise SystemExit(f"runtime_trace_receipt latest should skip malformed message rows: {hostile_trace}")
                hostile_metadata = hostile_trace.metadata
                if hostile_metadata.get("message_id") != baseline_latest_metadata.get("message_id"):
                    raise SystemExit(f"runtime_trace_receipt latest chose wrong readable trace: {hostile_metadata}")
                if hostile_metadata.get("readable_message_rows") != len(readable_messages):
                    raise SystemExit(f"runtime_trace_receipt latest missed readable message count: {hostile_metadata}")
                if hostile_metadata.get("unreadable_message_rows") != 1:
                    raise SystemExit(f"runtime_trace_receipt latest missed unreadable message count: {hostile_metadata}")
                hostile_handoff = hostile_metadata.get("runtime_trace_receipt_handoff") or {}
                if (hostile_handoff.get("counts") or {}).get("unreadable_message_rows") != 1:
                    raise SystemExit(f"runtime_trace_receipt handoff missed unreadable message count: {hostile_metadata}")
                if "unreadable message row(s) hidden for safety" not in hostile_trace.output:
                    raise SystemExit(f"runtime_trace_receipt latest should report hidden message rows: {hostile_trace.output}")
                hostile_payload = hostile_trace.output + json.dumps(hostile_metadata, sort_keys=True, default=str)
                if marker in hostile_payload:
                    raise SystemExit("runtime_trace_receipt latest leaked raw malformed message row text.")

                original_get_message = runtime.store.get_message

                def hostile_get_message(_message_id: int):
                    return HostileRow(marker)

                try:
                    runtime.store.get_message = hostile_get_message  # type: ignore[method-assign]
                    hostile_exact = runtime_trace_receipt({"message_id": metadata.get("message_id")})
                finally:
                    runtime.store.get_message = original_get_message  # type: ignore[method-assign]
                if hostile_exact.ok or hostile_exact.metadata.get("verdict") != "UNREADABLE_MESSAGE":
                    raise SystemExit(f"runtime_trace_receipt exact unreadable message should fail closed: {hostile_exact}")
                if hostile_exact.metadata.get("unreadable_message_rows", 0) < 1:
                    raise SystemExit(f"runtime_trace_receipt exact missed unreadable metadata: {hostile_exact.metadata}")
                hostile_exact_payload = hostile_exact.output + json.dumps(hostile_exact.metadata, sort_keys=True, default=str)
                if marker in hostile_exact_payload:
                    raise SystemExit("runtime_trace_receipt exact leaked raw malformed message row text.")
            if case in {"execution recovery", "what should Jarvis do after the failed run"}:
                for expected in [
                    "Jarvis execution recovery packet",
                    "read-only recovery steering layer",
                    "Problem run:",
                    "Verdict: HELD_FOR_APPROVAL",
                    "toolset: code",
                    "Recovery kind: last_look_approval_review",
                    "Safe recovery sequence",
                    "approval packet 1",
                    "approval chain proof 1",
                    "verification receipt <approved run id from approval chain proof 1>",
                    "Approval proof chain handoff",
                    "Learning hook",
                    "Next command queue",
                    "Recovery closure checklist",
                    "Retry readiness: blocked",
                    "Stop conditions",
                    "explicit stop times",
                    "does not retry tools",
                ]:
                    if expected not in result.response:
                        raise SystemExit(f"execution recovery packet missing expected text: {expected}")
                metadata = result.tool_results[0].metadata
                if metadata.get("verdict") != "HELD_FOR_APPROVAL" or metadata.get("problem_found") is not True:
                    raise SystemExit(f"execution recovery packet missed held approval metadata: {metadata}")
                if metadata.get("toolset") != "code":
                    raise SystemExit(f"execution recovery packet missed toolset metadata: {metadata}")
                if 1 not in metadata.get("recovery_approval_ids", []):
                    raise SystemExit(f"execution recovery packet missed recovery approval id: {metadata}")
                for expected_command in [
                    "verification receipt 2",
                    "approval readiness 1",
                    "approval packet 1",
                    "approval chain proof 1",
                    "verification receipt <approved run id from approval chain proof 1>",
                    "execution learning closure 2",
                    "after-action learning packet 2",
                    "execution health report",
                ]:
                    if expected_command not in metadata.get("next_commands", []):
                        raise SystemExit(f"execution recovery packet missed queued next command {expected_command}: {metadata}")
                expected_chain = [
                    "approval readiness 1",
                    "approval packet 1",
                    "approve approval 1",
                    "approval chain proof 1",
                    "verification receipt <approved run id from approval chain proof 1>",
                ]
                if metadata.get("approval_proof_chains", {}).get("1") != expected_chain:
                    raise SystemExit(f"execution recovery packet missed approval proof-chain metadata: {metadata}")
                if metadata.get("next_command") != "verification receipt 2" or metadata.get("next_command_count", 0) < 5:
                    raise SystemExit(f"execution recovery packet missed ordered next-command metadata: {metadata}")
                if metadata.get("recovery_closure_blocks_retry") is not True:
                    raise SystemExit(f"execution recovery packet missed recovery-closure retry block: {metadata}")
                if metadata.get("recovery_closure_next_required_command") != "verification receipt 2":
                    raise SystemExit(f"execution recovery packet missed first recovery-closure command: {metadata}")
                closure_commands = metadata.get("recovery_closure_required_commands", [])
                if metadata.get("recovery_closure_proof_queue") != closure_commands:
                    raise SystemExit(f"execution recovery packet proof queue should mirror required commands: {metadata}")
                if metadata.get("recovery_closure_proof_queue_count") != len(closure_commands):
                    raise SystemExit(f"execution recovery packet proof queue count diverged: {metadata}")
                if metadata.get("recovery_closure_next_proof_command") != "verification receipt 2":
                    raise SystemExit(f"execution recovery packet missed next recovery proof command: {metadata}")
                for expected_command in [
                    "verification receipt 2",
                    "execution recovery packet 2",
                    "after-action learning packet 2",
                    "execution learning closure 2",
                    "execution audit gate",
                    "approval chain proof 1",
                ]:
                    if expected_command not in metadata.get("recovery_closure_required_commands", []):
                        raise SystemExit(f"execution recovery packet missed closure required command {expected_command}: {metadata}")
                if closure_commands.index("after-action learning packet 2") > closure_commands.index("execution learning closure 2"):
                    raise SystemExit(f"execution recovery packet should place after-action learning before closure: {metadata}")
                closure_names = {str(check.get("name")) for check in metadata.get("recovery_closure_checks", [])}
                for expected_name in ["verification receipt", "recovery packet", "execution learning closure", "after-action learning", "execution audit gate", "approval proof chain 1"]:
                    if expected_name not in closure_names:
                        raise SystemExit(f"execution recovery packet missed closure check {expected_name}: {metadata}")
                closure_check_names = [str(check.get("name")) for check in metadata.get("recovery_closure_checks", [])]
                if closure_check_names.index("after-action learning") > closure_check_names.index("execution learning closure"):
                    raise SystemExit(f"execution recovery packet closure checks should place after-action learning before closure: {metadata}")
                handoff = metadata.get("execution_recovery_handoff") or {}
                if handoff.get("source") != "execution_recovery_packet":
                    raise SystemExit(f"execution recovery packet missed structured handoff: {metadata}")
                for nested_key, flat_key in {
                    "verdict": "verdict",
                    "recovery_kind": "recovery_kind",
                    "problem_found": "problem_found",
                    "next_command": "next_command",
                    "next_commands": "next_commands",
                    "next_command_count": "next_command_count",
                    "failure_to_test_command": "failure_to_test_command",
                }.items():
                    if handoff.get(nested_key) != metadata.get(flat_key):
                        raise SystemExit(f"execution recovery handoff field {nested_key} diverged from {flat_key}: {metadata}")
                target = handoff.get("target") or {}
                for nested_key, flat_key in {
                    "run_id": "run_id",
                    "tool_name": "tool_name",
                    "risk": "risk",
                    "toolset": "toolset",
                    "ok": "ok",
                    "approved": "approved",
                    "approval_id": "approval_id",
                    "failure_kind": "failure_kind",
                    "executed_handler": "target_executed_handler",
                    "planned_arg_keys": "planned_arg_keys",
                }.items():
                    if target.get(nested_key) != metadata.get(flat_key):
                        raise SystemExit(f"execution recovery target handoff field {nested_key} diverged: {metadata}")
                if (
                    metadata.get("executed_handler") is not True
                    or metadata.get("handler_invoked") is not True
                ):
                    raise SystemExit(
                        "execution recovery root invocation truth must remain executor-owned: "
                        f"{metadata}"
                    )
                approval = handoff.get("approval") or {}
                for nested_key, flat_key in {
                    "risk_needs_approval": "risk_needs_approval",
                    "output_approval_ids": "output_approval_ids",
                    "inferred_approval_ids": "inferred_approval_ids",
                    "recovery_approval_ids": "recovery_approval_ids",
                    "proof_chains": "approval_proof_chains",
                }.items():
                    if approval.get(nested_key) != metadata.get(flat_key):
                        raise SystemExit(f"execution recovery approval handoff field {nested_key} diverged: {metadata}")
                recovery = handoff.get("recovery_closure") or {}
                for nested_key, flat_key in {
                    "checks": "recovery_closure_checks",
                    "check_count": "recovery_closure_check_count",
                    "required_commands": "recovery_closure_required_commands",
                    "required_command_count": "recovery_closure_proof_queue_count",
                    "proof_queue": "recovery_closure_proof_queue",
                    "proof_queue_count": "recovery_closure_proof_queue_count",
                    "next_required_command": "recovery_closure_next_required_command",
                    "next_proof_command": "recovery_closure_next_proof_command",
                    "blocks_retry": "recovery_closure_blocks_retry",
                }.items():
                    if recovery.get(nested_key) != metadata.get(flat_key):
                        raise SystemExit(f"execution recovery closure handoff field {nested_key} diverged: {metadata}")
                for key in ["review_only", "draft_only", "loads_without_execution"]:
                    if handoff.get(key) is not True or metadata.get(f"execution_recovery_{key}") is not True:
                        raise SystemExit(f"execution recovery handoff missed true {key}: {metadata}")
                for key in [
                    "authorizes_execution",
                    "authorizes_completion_claim",
                    "approval_granted",
                    "calls_model",
                    "executes_tools",
                    "writes_files",
                    "reads_personal_data",
                    "external_side_effect",
                    "controls_computer",
                    "queues_approval",
                ]:
                    if handoff.get(key) is not False:
                        raise SystemExit(f"execution recovery handoff should keep {key}=False: {metadata}")
                if (
                    metadata.get("execution_recovery_authorizes_execution") is not False
                    or metadata.get("execution_recovery_authorizes_completion_claim") is not False
                    or metadata.get("execution_recovery_approval_granted") is not False
                ):
                    raise SystemExit(f"execution recovery flat authority fields should be false: {metadata}")
                assert_no_future_authority(metadata, "execution recovery")
                if metadata.get("operator_timeboxes_override_priority") is not True or metadata.get("stop_times_override_priority") is not True:
                    raise SystemExit(f"execution recovery packet missed operator-limit metadata: {metadata}")
                if (
                    metadata.get("executes_tools")
                    or metadata.get("queues_approval")
                    or metadata.get("approves_request")
                    or metadata.get("dismisses_request")
                    or metadata.get("reads_private_data")
                    or metadata.get("writes_files")
                    or metadata.get("controls_computer")
                ):
                    raise SystemExit("execution recovery packet should remain read-only.")
            if case in {"after-action learning", "what can Jarvis learn from run 2"}:
                for expected in [
                    "Jarvis after-action learning packet",
                    "read-only learning layer after execution",
                    "Source run:",
                    "run_shell_command",
                    "toolset: code",
                    "Verdict:",
                    "Learning candidates:",
                    "Safe review commands:",
                    "Next command queue",
                    "Learning proof queue:",
                    "next required command:",
                    "proof queue count:",
                    "proof queue:",
                    "verification receipt",
                    "execution recovery packet",
                    "Approval proof chain handoff",
                    "approval chain proof 1",
                    "verification receipt <approved run id from approval chain proof 1>",
                    "failure to test preview:",
                    "Promotion rules:",
                    "explicit stop times",
                    "does not save memories",
                ]:
                    if expected not in result.response:
                        raise SystemExit(f"after_action_learning_packet missing expected text: {expected}")
                if "Learning proof queue:\n- next proof command:" in result.response:
                    raise SystemExit("after_action_learning_packet should render learning queue head as next required command.")
                metadata = result.tool_results[0].metadata
                if metadata.get("learning_candidates", 0) < 1:
                    raise SystemExit(f"after_action_learning_packet missed learning candidates: {metadata}")
                if metadata.get("run_id") != 2 or metadata.get("tool_name") != "run_shell_command":
                    raise SystemExit(f"after_action_learning_packet should skip meta audit packets and inspect the problem action: {metadata}")
                if metadata.get("toolset") != "code":
                    raise SystemExit(f"after_action_learning_packet missed toolset metadata: {metadata}")
                if not str(metadata.get("verification_command", "")).startswith("verification receipt "):
                    raise SystemExit(f"after_action_learning_packet missed verification command: {metadata}")
                for expected_command in [
                    "verification receipt 2",
                    "execution recovery packet 2",
                    "approval readiness 1",
                    "approval packet 1",
                    "approval chain proof 1",
                    "verification receipt <approved run id from approval chain proof 1>",
                    "learning review",
                ]:
                    if expected_command not in metadata.get("next_commands", []):
                        raise SystemExit(f"after_action_learning_packet missed queued next command {expected_command}: {metadata}")
                if metadata.get("approval_proof_chains", {}).get("1") != [
                    "approval readiness 1",
                    "approval packet 1",
                    "approve approval 1",
                    "approval chain proof 1",
                    "verification receipt <approved run id from approval chain proof 1>",
                ]:
                    raise SystemExit(f"after_action_learning_packet missed approval proof-chain metadata: {metadata}")
                if metadata.get("next_command") != "verification receipt 2" or metadata.get("next_command_count", 0) < 5:
                    raise SystemExit(f"after_action_learning_packet missed ordered next-command metadata: {metadata}")
                if metadata.get("next_required_command") != "verification receipt 2":
                    raise SystemExit(f"after_action_learning_packet missed next required command: {metadata}")
                proof_queue = metadata.get("proof_queue", [])
                if not proof_queue or proof_queue != metadata.get("next_commands", []):
                    raise SystemExit(f"after_action_learning_packet missed proof queue alias: {metadata}")
                if metadata.get("proof_queue_count") != len(proof_queue):
                    raise SystemExit(f"after_action_learning_packet missed proof queue count: {metadata}")
                if metadata.get("next_proof_command") != proof_queue[0]:
                    raise SystemExit(f"after_action_learning_packet missed next proof command: {metadata}")
                if metadata.get("learning_proof_queue") != proof_queue or metadata.get("after_action_learning_proof_queue") != proof_queue:
                    raise SystemExit(f"after_action_learning_packet missed learning proof queue aliases: {metadata}")
                if metadata.get("learning_proof_queue_count") != len(proof_queue) or metadata.get("after_action_learning_proof_queue_count") != len(proof_queue):
                    raise SystemExit(f"after_action_learning_packet missed learning proof queue alias counts: {metadata}")
                if metadata.get("learning_next_proof_command") != proof_queue[0] or metadata.get("after_action_learning_next_proof_command") != proof_queue[0]:
                    raise SystemExit(f"after_action_learning_packet missed learning next proof aliases: {metadata}")
                if metadata.get("learning_next_required_command") != proof_queue[0] or metadata.get("after_action_learning_next_required_command") != proof_queue[0]:
                    raise SystemExit(f"after_action_learning_packet missed learning next required aliases: {metadata}")
                handoff = metadata.get("after_action_learning_handoff") or {}
                if handoff.get("source") != "after_action_learning_packet":
                    raise SystemExit(f"after_action_learning_packet missed handoff source: {metadata}")
                for nested_key, flat_key in {
                    "verdict": "verdict",
                    "next_command": "next_command",
                    "next_required_command": "next_required_command",
                    "next_commands": "next_commands",
                    "next_command_count": "next_command_count",
                    "verification_command": "verification_command",
                    "recovery_command": "recovery_command",
                    "regression_command": "regression_command",
                    "task_command": "task_command",
                    "proof_queue": "proof_queue",
                    "proof_queue_count": "proof_queue_count",
                    "next_proof_command": "next_proof_command",
                    "learning_proof_queue": "learning_proof_queue",
                    "learning_proof_queue_count": "learning_proof_queue_count",
                    "learning_next_required_command": "learning_next_required_command",
                    "learning_next_proof_command": "learning_next_proof_command",
                    "after_action_learning_proof_queue": "after_action_learning_proof_queue",
                    "after_action_learning_proof_queue_count": "after_action_learning_proof_queue_count",
                    "after_action_learning_next_required_command": "after_action_learning_next_required_command",
                    "after_action_learning_next_proof_command": "after_action_learning_next_proof_command",
                }.items():
                    if handoff.get(nested_key) != metadata.get(flat_key):
                        raise SystemExit(f"after_action_learning handoff field {nested_key} diverged from {flat_key}: {metadata}")
                if handoff.get("learning_candidates") != metadata.get("learning_candidate_names"):
                    raise SystemExit(f"after_action_learning handoff candidate names diverged: {metadata}")
                if handoff.get("learning_candidate_count") != metadata.get("learning_candidates"):
                    raise SystemExit(f"after_action_learning handoff candidate count diverged: {metadata}")
                if handoff.get("promotion_allowed") != (metadata.get("verdict") == "SAFE_TO_REVIEW_FOR_LEARNING"):
                    raise SystemExit(f"after_action_learning handoff promotion flag diverged: {metadata}")
                target = handoff.get("target") or {}
                for nested_key, flat_key in {
                    "run_id": "run_id",
                    "tool_name": "tool_name",
                    "risk": "risk",
                    "toolset": "toolset",
                    "ok": "ok",
                    "approved": "approved",
                    "approval_id": "approval_id",
                    "failure_kind": "failure_kind",
                    "planned_arg_keys": "planned_arg_keys",
                }.items():
                    if target.get(nested_key) != metadata.get(flat_key):
                        raise SystemExit(f"after_action_learning target handoff field {nested_key} diverged: {metadata}")
                approval = handoff.get("approval") or {}
                for nested_key, flat_key in {
                    "risk_needs_approval": "risk_needs_approval",
                    "approval_problem": "approval_problem",
                    "output_approval_ids": "output_approval_ids",
                    "approval_proof_ids": "approval_proof_ids",
                    "proof_chains": "approval_proof_chains",
                }.items():
                    if approval.get(nested_key) != metadata.get(flat_key):
                        raise SystemExit(f"after_action_learning approval handoff field {nested_key} diverged: {metadata}")
                for key in ["review_only", "draft_only", "loads_without_execution"]:
                    if handoff.get(key) is not True or metadata.get(f"after_action_learning_{key}") is not True:
                        raise SystemExit(f"after_action_learning handoff missed true {key}: {metadata}")
                for key in [
                    "authorizes_execution",
                    "authorizes_completion_claim",
                    "approval_granted",
                    "calls_model",
                    "executes_tools",
                    "writes_files",
                    "reads_personal_data",
                    "external_side_effect",
                    "controls_computer",
                    "queues_approval",
                ]:
                    if handoff.get(key) is not False:
                        raise SystemExit(f"after_action_learning handoff should keep {key}=False: {metadata}")
                if (
                    metadata.get("after_action_learning_authorizes_execution") is not False
                    or metadata.get("after_action_learning_authorizes_completion_claim") is not False
                    or metadata.get("after_action_learning_approval_granted") is not False
                ):
                    raise SystemExit(f"after_action_learning flat authority fields should be false: {metadata}")
                assert_no_future_authority(metadata, "after-action learning")
                if metadata.get("operator_timeboxes_override_priority") is not True or metadata.get("stop_times_override_priority") is not True:
                    raise SystemExit(f"after_action_learning_packet missed operator-limit metadata: {metadata}")
                if (
                    metadata.get("executes_tools")
                    or metadata.get("queues_approval")
                    or metadata.get("approves_request")
                    or metadata.get("dismisses_request")
                    or metadata.get("reads_private_data")
                    or metadata.get("writes_files")
                    or metadata.get("writes_notes")
                    or metadata.get("writes_memory")
                    or metadata.get("controls_computer")
                ):
                    raise SystemExit("after_action_learning_packet should remain read-only.")
            if case in {"execution health report", "is execution healthy"}:
                for expected in [
                    "Jarvis execution health report",
                    "aggregate runtime health layer",
                    "Verdict:",
                    "Action runs:",
                    "Verification/audit packets:",
                    "Blocker categories:",
                    "Verification coverage:",
                    "Runtime queue ledger:",
                    "runtime traces inspected:",
                    "latest queue delta:",
                    "total queue delta:",
                    "new approval ids: 1",
                    "referenced approval ids: 1",
                    "Verification-to-learning handoff:",
                    "Recovery closure gate:",
                    "ready for operator retry review:",
                    "Operator limits still apply",
                    "target run:",
                    "target toolset: code",
                    "verify:",
                    "recover:",
                    "learn:",
                    "command-first rule:",
                    "Repeated failure tools:",
                    "Failure promotion queue:",
                    "Newest action signals:",
                    "Next safe command:",
                    "Execution proof queue:",
                    "next required command:",
                    "proof queue count:",
                    "proof queue:",
                    "Recovery queue:",
                    "does not call models",
                ]:
                    if expected not in result.response:
                        raise SystemExit(f"execution_health_report missing expected text: {expected}")
                if "Execution proof queue:\n- next proof command:" in result.response:
                    raise SystemExit("execution_health_report should render execution queue head as next required command.")
                metadata = result.tool_results[0].metadata
                if metadata.get("inspected_runs", 0) < 1 or not str(metadata.get("verdict", "")).strip():
                    raise SystemExit(f"execution_health_report missed health metadata: {metadata}")
                if metadata.get("action_runs", 0) < 1 or metadata.get("failed_or_blocked_runs", 0) < 1:
                    raise SystemExit(f"execution_health_report missed action/failure counts: {metadata}")
                if not str(metadata.get("next_command", "")).strip():
                    raise SystemExit(f"execution_health_report missed next command: {metadata}")
                if metadata.get("next_command") not in metadata.get("next_commands", []):
                    raise SystemExit(f"execution_health_report should include primary next command in recovery queue: {metadata}")
                if metadata.get("next_command_count", 0) < 2:
                    raise SystemExit(f"execution_health_report should expose an ordered recovery queue: {metadata}")
                proof_queue = metadata.get("execution_proof_queue", [])
                if not proof_queue or metadata.get("proof_queue") != proof_queue:
                    raise SystemExit(f"execution_health_report missed proof queue metadata: {metadata}")
                if metadata.get("execution_proof_queue_count") != len(proof_queue) or metadata.get("proof_queue_count") != len(proof_queue):
                    raise SystemExit(f"execution_health_report missed proof queue count metadata: {metadata}")
                if metadata.get("execution_next_proof_command") != proof_queue[0] or metadata.get("next_proof_command") != proof_queue[0]:
                    raise SystemExit(f"execution_health_report missed next proof command metadata: {metadata}")
                if metadata.get("execution_next_required_command") != proof_queue[0] or metadata.get("next_required_command") != proof_queue[0]:
                    raise SystemExit(f"execution_health_report missed next required command metadata: {metadata}")
                if metadata.get("next_commands", [])[: len(proof_queue)] != proof_queue:
                    raise SystemExit(f"execution_health_report proof queue should mirror ordered next commands: {metadata}")
                handoff = metadata.get("execution_health_handoff") or {}
                if handoff.get("source") != "execution_health_report":
                    raise SystemExit(f"execution_health_report missed structured health handoff: {metadata}")
                for nested_key, flat_key in {
                    "verdict": "verdict",
                    "review_required": "review_required",
                    "safe_to_continue": "safe_to_continue",
                    "inspected_runs": "inspected_runs",
                    "action_runs": "action_runs",
                    "failed_or_blocked_runs": "failed_or_blocked_runs",
                    "risky_approval_problems": "risky_approval_problems",
                    "verification_coverage_state": "verification_coverage_state",
                    "blocker_categories": "blocker_categories",
                    "blocker_count": "blocker_count",
                    "next_command": "next_command",
                    "next_commands": "next_commands",
                    "next_command_count": "next_command_count",
                    "proof_queue": "execution_proof_queue",
                    "proof_queue_count": "execution_proof_queue_count",
                    "next_proof_command": "next_proof_command",
                    "learning_followup_command": "learning_followup_command",
                }.items():
                    if handoff.get(nested_key) != metadata.get(flat_key):
                        raise SystemExit(f"execution_health_report handoff field {nested_key} diverged from {flat_key}: {metadata}")
                learning_target = handoff.get("learning_target") or {}
                for nested_key, flat_key in {
                    "run_id": "learning_target_run_id",
                    "tool": "learning_target_tool",
                    "toolset": "learning_target_toolset",
                    "verification_handoff_command": "verification_handoff_command",
                    "recovery_handoff_command": "recovery_handoff_command",
                    "learning_closure_command": "learning_closure_command",
                    "learning_handoff_command": "learning_handoff_command",
                }.items():
                    if learning_target.get(nested_key) != metadata.get(flat_key):
                        raise SystemExit(f"execution_health_report learning handoff field {nested_key} diverged: {metadata}")
                recovery = handoff.get("recovery_closure") or {}
                for nested_key, flat_key in {
                    "state": "recovery_closure_state",
                    "missing": "recovery_closure_missing",
                    "missing_count": "recovery_closure_missing_count",
                    "proof_queue": "recovery_closure_proof_queue",
                    "proof_queue_count": "recovery_closure_proof_queue_count",
                    "next_proof_command": "recovery_closure_next_proof_command",
                    "ready_to_retry": "recovery_closure_ready_to_retry",
                    "blocks_auto_execution": "recovery_closure_blocks_auto_execution",
                    "blocks_completion_claim": "recovery_closure_blocks_completion_claim",
                    "target_run_id": "recovery_closure_target_run_id",
                    "target_tool_name": "recovery_closure_target_tool_name",
                    "target_toolset": "recovery_closure_target_toolset",
                    "target_verification_receipts": "recovery_closure_target_verification_receipts",
                    "target_recovery_packets": "recovery_closure_target_recovery_packets",
                    "target_after_action_learning_packets": "recovery_closure_target_after_action_learning_packets",
                }.items():
                    if recovery.get(nested_key) != metadata.get(flat_key):
                        raise SystemExit(f"execution_health_report recovery handoff field {nested_key} diverged: {metadata}")
                failure_promotion = handoff.get("failure_promotion") or {}
                if failure_promotion.get("queue") != metadata.get("failure_promotion_queue"):
                    raise SystemExit(f"execution_health_report failure promotion handoff queue diverged: {metadata}")
                if failure_promotion.get("queue_count") != metadata.get("failure_promotion_queue_count"):
                    raise SystemExit(f"execution_health_report failure promotion handoff count diverged: {metadata}")
                for key in ["review_only", "draft_only", "loads_without_execution"]:
                    if handoff.get(key) is not True or metadata.get(f"execution_health_{key}") is not True:
                        raise SystemExit(f"execution_health_report handoff missed true {key}: {metadata}")
                for key in [
                    "authorizes_execution",
                    "authorizes_completion_claim",
                    "approval_granted",
                    "calls_model",
                    "executes_tools",
                    "writes_files",
                    "reads_personal_data",
                    "external_side_effect",
                    "controls_computer",
                    "queues_approval",
                ]:
                    if handoff.get(key) is not False:
                        raise SystemExit(f"execution_health_report handoff should keep {key}=False: {metadata}")
                if metadata.get("execution_health_authorizes_execution") is not False or metadata.get("execution_health_authorizes_completion_claim") is not False:
                    raise SystemExit(f"execution_health_report flat health authority fields should be false: {metadata}")
                if metadata.get("execution_health_approval_granted") is not False:
                    raise SystemExit(f"execution_health_report flat health approval should be false: {metadata}")
                if metadata.get("operator_timeboxes_override_priority") is not True or metadata.get("stop_times_override_priority") is not True:
                    raise SystemExit(f"execution_health_report missed operator-limit metadata: {metadata}")
                for expected_category in ["failed_or_blocked"]:
                    if expected_category not in metadata.get("blocker_categories", []):
                        raise SystemExit(f"execution_health_report missed blocker category {expected_category}: {metadata}")
                if metadata.get("blocker_count", 0) < 1:
                    raise SystemExit(f"execution_health_report missed blocker count: {metadata}")
                if metadata.get("verification_coverage_state") not in {"present", "missing"}:
                    raise SystemExit(f"execution_health_report missed verification coverage state: {metadata}")
                if metadata.get("runtime_trace_count", 0) < 1:
                    raise SystemExit(f"execution_health_report missed runtime trace count: {metadata}")
                if 1 not in metadata.get("runtime_queued_approval_ids", []):
                    raise SystemExit(f"execution_health_report missed queued approval id from runtime traces: {metadata}")
                if 1 not in metadata.get("runtime_new_approval_ids", []):
                    raise SystemExit(f"execution_health_report missed new approval id from runtime traces: {metadata}")
                if 1 not in metadata.get("runtime_referenced_approval_ids", []):
                    raise SystemExit(f"execution_health_report missed referenced approval id from runtime traces: {metadata}")
                if metadata.get("latest_approval_queue_delta") != 0:
                    raise SystemExit(f"execution_health_report should show the latest reporting turn was approval-neutral: {metadata}")
                if metadata.get("runtime_queue_delta_total") != 0:
                    raise SystemExit(f"execution_health_report should balance queued and approved approval movement in this smoke: {metadata}")
                if "failure_promotion_queue" not in metadata or "failure_promotion_queue_count" not in metadata:
                    raise SystemExit(f"execution_health_report missed failure promotion queue metadata: {metadata}")
                if metadata.get("learning_target_run_id") != 2 or metadata.get("learning_target_tool") != "run_shell_command":
                    raise SystemExit(f"execution_health_report missed learning target handoff: {metadata}")
                if metadata.get("learning_target_toolset") != "code" or metadata.get("recovery_closure_target_toolset") != "code":
                    raise SystemExit(f"execution_health_report missed target toolset handoff: {metadata}")
                if metadata.get("verification_handoff_command") != "verification receipt 2":
                    raise SystemExit(f"execution_health_report missed verification handoff command: {metadata}")
                if metadata.get("recovery_handoff_command") != "execution recovery packet 2":
                    raise SystemExit(f"execution_health_report missed recovery handoff command: {metadata}")
                if metadata.get("learning_handoff_command") != "after-action learning packet 2":
                    raise SystemExit(f"execution_health_report missed learning handoff command: {metadata}")
                if metadata.get("learning_followup_command") != "after-action learning packet 2":
                    raise SystemExit(f"execution_health_report missed learning follow-up command: {metadata}")
                if metadata.get("learning_closure_command") != "execution learning closure 2":
                    raise SystemExit(f"execution_health_report missed learning closure command: {metadata}")
                if metadata.get("execution_learning_closure_command") != "execution learning closure 2":
                    raise SystemExit(f"execution_health_report missed execution learning closure command: {metadata}")
                if metadata.get("recovery_closure_state") != "blocked_missing_target_verification_receipt":
                    raise SystemExit(f"execution_health_report missed recovery closure state: {metadata}")
                if metadata.get("recovery_closure_ready_to_retry") is not False:
                    raise SystemExit(f"execution_health_report should block retry until closure evidence exists: {metadata}")
                if metadata.get("recovery_closure_blocks_auto_execution") is not True:
                    raise SystemExit(f"execution_health_report should block auto-run until closure evidence exists: {metadata}")
                if metadata.get("target_recovery_packets", 0) < 1 or metadata.get("target_after_action_learning_packets", 0) < 1:
                    raise SystemExit(f"execution_health_report missed target recovery/learning packet counts: {metadata}")
                if metadata.get("target_verification_receipts", 0) != 0:
                    raise SystemExit(f"execution_health_report should report missing target verification receipt: {metadata}")
                if "target_verification_receipt" not in metadata.get("recovery_closure_missing", []):
                    raise SystemExit(f"execution_health_report missed closure missing reason: {metadata}")
                if "verification receipt 2" not in metadata.get("recovery_closure_required_commands", []):
                    raise SystemExit(f"execution_health_report missed closure required command: {metadata}")
                closure_commands = metadata.get("recovery_closure_required_commands", [])
                if metadata.get("recovery_closure_next_required_command") != closure_commands[0]:
                    raise SystemExit(f"execution_health_report missed explicit first recovery proof command: {metadata}")
                if metadata.get("recovery_closure_proof_queue") != closure_commands:
                    raise SystemExit(f"execution_health_report recovery proof queue should mirror required commands: {metadata}")
                if metadata.get("recovery_closure_proof_queue_count") != len(closure_commands):
                    raise SystemExit(f"execution_health_report recovery proof queue count diverged: {metadata}")
                if metadata.get("recovery_closure_next_proof_command") != closure_commands[0]:
                    raise SystemExit(f"execution_health_report missed next recovery proof command: {metadata}")
                if metadata.get("next_command") != closure_commands[0]:
                    raise SystemExit(f"execution_health_report should route next command to first recovery-closure proof: {metadata}")
                if metadata.get("next_commands", [])[: len(closure_commands)] != closure_commands:
                    raise SystemExit(f"execution_health_report should put closure proofs first in the queue: {metadata}")
                for expected_command in [
                    "verification receipt 2",
                    "execution recovery packet 2",
                    "execution learning closure 2",
                    "after-action learning packet 2",
                ]:
                    if expected_command not in metadata.get("next_commands", []):
                        raise SystemExit(f"execution_health_report missed handoff command in recovery queue: {expected_command} in {metadata}")
                if (
                    metadata.get("executes_tools")
                    or metadata.get("queues_approval")
                    or metadata.get("approves_request")
                    or metadata.get("dismisses_request")
                    or metadata.get("reads_private_data")
                    or metadata.get("writes_files")
                    or metadata.get("writes_notes")
                    or metadata.get("writes_memory")
                    or metadata.get("controls_computer")
                ):
                    raise SystemExit("execution_health_report should remain read-only.")
            if case == "recovery closure checklist":
                for expected in [
                    "Jarvis recovery closure checklist",
                    "operator checklist",
                    "Closure state:",
                    "Target run:",
                    "Target toolset: code",
                    "Next closure command:",
                    "Checklist rows:",
                    "target verification:",
                    "target recovery:",
                    "after-action learning:",
                    "approval chain:",
                    "Required closure queue:",
                    "does not execute",
                ]:
                    if expected not in result.response:
                        raise SystemExit(f"recovery_closure_checklist missing expected text: {expected}")
                metadata = result.tool_results[0].metadata
                if metadata.get("closure_state") != "blocked_missing_target_verification_receipt":
                    raise SystemExit(f"recovery_closure_checklist missed closure state: {metadata}")
                if metadata.get("target_run_id") != 2 or metadata.get("target_tool_name") != "run_shell_command":
                    raise SystemExit(f"recovery_closure_checklist missed target run: {metadata}")
                if metadata.get("target_toolset") != "code" or metadata.get("recovery_closure_target_toolset") != "code":
                    raise SystemExit(f"recovery_closure_checklist missed target toolset: {metadata}")
                if metadata.get("ready_to_retry") is not False or metadata.get("blocks_auto_execution") is not True:
                    raise SystemExit(f"recovery_closure_checklist missed retry block: {metadata}")
                if "target_verification_receipt" not in metadata.get("missing", []):
                    raise SystemExit(f"recovery_closure_checklist missed missing verification proof: {metadata}")
                if "verification receipt 2" not in metadata.get("required_commands", []):
                    raise SystemExit(f"recovery_closure_checklist missed required command: {metadata}")
                if metadata.get("next_command") != "verification receipt 2" or metadata.get("next_proof_command") != "verification receipt 2":
                    raise SystemExit(f"recovery_closure_checklist missed next proof command: {metadata}")
                rows = metadata.get("checklist_rows", [])
                if metadata.get("checklist_row_count") != len(rows) or len(rows) < 4:
                    raise SystemExit(f"recovery_closure_checklist missed checklist rows: {metadata}")
                if not any(row.get("phase") == "target verification" and row.get("state") == "missing" for row in rows):
                    raise SystemExit(f"recovery_closure_checklist missed target verification row: {metadata}")
                if metadata.get("target_recovery_packets", 0) < 1 or metadata.get("target_after_action_learning_packets", 0) < 1:
                    raise SystemExit(f"recovery_closure_checklist missed target proof counts: {metadata}")
                handoff = metadata.get("recovery_closure_handoff") or {}
                if handoff.get("source") != "recovery_closure_checklist":
                    raise SystemExit(f"recovery_closure_checklist missed structured handoff: {metadata}")
                for nested_key, flat_key in {
                    "source_tool": "source_tool",
                    "closure_state": "closure_state",
                    "target_run_id": "target_run_id",
                    "target_tool_name": "target_tool_name",
                    "target_toolset": "target_toolset",
                    "ready_to_retry": "ready_to_retry",
                    "blocks_auto_execution": "blocks_auto_execution",
                    "blocks_completion_claim": "blocks_completion_claim",
                    "missing": "missing",
                    "missing_count": "missing_count",
                    "required_commands": "required_commands",
                    "required_command_count": "required_command_count",
                    "next_command": "next_command",
                    "proof_queue": "proof_queue",
                    "proof_queue_count": "proof_queue_count",
                    "next_proof_command": "next_proof_command",
                    "checklist_rows": "checklist_rows",
                    "checklist_row_count": "checklist_row_count",
                    "target_verification_receipts": "target_verification_receipts",
                    "target_recovery_packets": "target_recovery_packets",
                    "target_after_action_learning_packets": "target_after_action_learning_packets",
                    "failure_promotion_queue": "failure_promotion_queue",
                    "failure_promotion_queue_count": "failure_promotion_queue_count",
                    "health_verdict": "health_verdict",
                    "inspected_runs": "inspected_runs",
                }.items():
                    if handoff.get(nested_key) != metadata.get(flat_key):
                        raise SystemExit(f"recovery_closure_checklist handoff field {nested_key} diverged from {flat_key}: {metadata}")
                for key in ["review_only", "draft_only", "loads_without_execution"]:
                    if handoff.get(key) is not True or metadata.get(f"recovery_closure_{key}") is not True:
                        raise SystemExit(f"recovery_closure_checklist handoff missed true {key}: {metadata}")
                for key in [
                    "authorizes_execution",
                    "authorizes_completion_claim",
                    "approval_granted",
                    "calls_model",
                    "executes_tools",
                    "writes_files",
                    "reads_personal_data",
                    "external_side_effect",
                    "controls_computer",
                    "queues_approval",
                ]:
                    if handoff.get(key) is not False:
                        raise SystemExit(f"recovery_closure_checklist handoff should keep {key}=False: {metadata}")
                if (
                    metadata.get("recovery_closure_authorizes_execution") is not False
                    or metadata.get("recovery_closure_authorizes_completion_claim") is not False
                    or metadata.get("recovery_closure_approval_granted") is not False
                ):
                    raise SystemExit(f"recovery_closure_checklist flat authority fields should be false: {metadata}")
                if (
                    metadata.get("executes_tools")
                    or metadata.get("queues_approval")
                    or metadata.get("approves_request")
                    or metadata.get("dismisses_request")
                    or metadata.get("reads_private_data")
                    or metadata.get("writes_files")
                    or metadata.get("writes_notes")
                    or metadata.get("writes_memory")
                    or metadata.get("controls_computer")
                ):
                    raise SystemExit("recovery_closure_checklist should remain read-only.")

            if case == "execution learning closure":
                for expected in [
                    "Jarvis execution learning closure packet",
                    "completion gate for the learning loop",
                    "Verdict: LEARNING_CLOSURE_INCOMPLETE",
                    "Target toolset: code",
                    "Learning closure checklist:",
                    "Target proof artifact ledger:",
                    "verification:",
                    "after-action learning:",
                    "repeated failure promotion:",
                    "Missing proof:",
                    "target_verification_receipt",
                    "Required learning proof queue:",
                    "verification receipt 2",
                    "Boundary:",
                ]:
                    if expected not in result.response:
                        raise SystemExit(f"execution_learning_closure_packet missing expected text: {expected}")
                metadata = result.tool_results[0].metadata
                if metadata.get("learning_closure_ready") is not False:
                    raise SystemExit(f"execution_learning_closure_packet should not be ready while verification is missing: {metadata}")
                if metadata.get("learning_closure_blocks_completion_claim") is not True:
                    raise SystemExit(f"execution_learning_closure_packet missed completion blocker flag: {metadata}")
                if metadata.get("target_run_id") != 2 or metadata.get("target_tool_name") != "run_shell_command":
                    raise SystemExit(f"execution_learning_closure_packet missed target run: {metadata}")
                if metadata.get("target_toolset") != "code":
                    raise SystemExit(f"execution_learning_closure_packet missed target toolset: {metadata}")
                if metadata.get("target_after_action_learning_packets", 0) < 1:
                    raise SystemExit(f"execution_learning_closure_packet missed target after-action proof count: {metadata}")
                if "target_verification_receipt" not in metadata.get("missing", []):
                    raise SystemExit(f"execution_learning_closure_packet missed missing verification proof: {metadata}")
                if "verification receipt 2" not in metadata.get("required_commands", []):
                    raise SystemExit(f"execution_learning_closure_packet missed required verification command: {metadata}")
                artifacts = metadata.get("target_proof_artifacts", {})
                if metadata.get("target_proof_artifact_count") != len(artifacts) or len(artifacts) < 5:
                    raise SystemExit(f"execution_learning_closure_packet missed proof artifact ledger: {metadata}")
                if metadata.get("target_verification_present") is not False:
                    raise SystemExit(f"execution_learning_closure_packet should mark target verification missing: {metadata}")
                if metadata.get("target_recovery_present") is not True:
                    raise SystemExit(f"execution_learning_closure_packet should mark target recovery present: {metadata}")
                if metadata.get("target_after_action_learning_present") is not True:
                    raise SystemExit(f"execution_learning_closure_packet should mark after-action learning present: {metadata}")
                if metadata.get("repeated_failure_promotion_present") is not True:
                    raise SystemExit(f"execution_learning_closure_packet should mark repeated failure promotion satisfied: {metadata}")
                if artifacts.get("verification", {}).get("command") != "verification receipt 2":
                    raise SystemExit(f"execution_learning_closure_packet missed verification artifact command: {metadata}")
                if artifacts.get("after_action_learning", {}).get("command") != "after-action learning packet 2":
                    raise SystemExit(f"execution_learning_closure_packet missed after-action artifact command: {metadata}")
                if metadata.get("next_command") != metadata.get("next_proof_command"):
                    raise SystemExit(f"execution_learning_closure_packet missed next proof alias: {metadata}")
                if metadata.get("next_required_command") != metadata.get("next_command"):
                    raise SystemExit(f"execution_learning_closure_packet missed next required alias: {metadata}")
                if metadata.get("execution_learning_closure_next_required_command") != metadata.get("next_command"):
                    raise SystemExit(f"execution_learning_closure_packet missed closure next required alias: {metadata}")
                if metadata.get("execution_learning_closure_next_proof_command") != metadata.get("next_command"):
                    raise SystemExit(f"execution_learning_closure_packet missed closure next proof alias: {metadata}")
                if metadata.get("proof_queue") != metadata.get("required_commands"):
                    raise SystemExit(f"execution_learning_closure_packet proof queue diverged from required commands: {metadata}")
                if metadata.get("execution_learning_closure_proof_queue") != metadata.get("proof_queue"):
                    raise SystemExit(f"execution_learning_closure_packet closure proof queue diverged: {metadata}")
                if metadata.get("actionable_next_required_command") != metadata.get("actionable_next_command"):
                    raise SystemExit(f"execution_learning_closure_packet missed actionable next required alias: {metadata}")
                if metadata.get("execution_learning_closure_actionable_next_required_command") != metadata.get("actionable_next_command"):
                    raise SystemExit(f"execution_learning_closure_packet missed closure actionable next required alias: {metadata}")
                if metadata.get("execution_learning_closure_actionable_next_proof_command") != metadata.get("actionable_next_command"):
                    raise SystemExit(f"execution_learning_closure_packet missed closure actionable next proof alias: {metadata}")
                rows = metadata.get("phase_rows", [])
                if metadata.get("phase_row_count") != len(rows) or len(rows) < 5:
                    raise SystemExit(f"execution_learning_closure_packet missed phase rows: {metadata}")
                if metadata.get("source_tool") != "execution_health_report":
                    raise SystemExit(f"execution_learning_closure_packet missed source tool: {metadata}")
                handoff = metadata.get("execution_learning_closure_handoff") or {}
                if handoff.get("source") != "execution_learning_closure_packet":
                    raise SystemExit(f"execution_learning_closure_packet missed structured handoff: {metadata}")
                for nested_key, flat_key in {
                    "source_tool": "source_tool",
                    "verdict": "verdict",
                    "health_verdict": "health_verdict",
                    "proof_artifacts": "target_proof_artifacts",
                    "proof_artifact_count": "target_proof_artifact_count",
                    "phase_rows": "phase_rows",
                    "phase_row_count": "phase_row_count",
                    "missing": "missing",
                    "missing_count": "missing_count",
                    "required_commands": "required_commands",
                    "required_command_count": "required_command_count",
                    "proof_queue": "proof_queue",
                    "proof_queue_count": "proof_queue_count",
                    "next_command": "next_command",
                    "next_required_command": "next_required_command",
                    "next_proof_command": "next_proof_command",
                    "execution_learning_closure_next_required_command": "execution_learning_closure_next_required_command",
                    "execution_learning_closure_next_proof_command": "execution_learning_closure_next_proof_command",
                    "actionable_required_commands": "actionable_required_commands",
                    "actionable_required_command_count": "actionable_required_command_count",
                    "actionable_proof_queue": "actionable_proof_queue",
                    "actionable_proof_queue_count": "actionable_proof_queue_count",
                    "actionable_next_command": "actionable_next_command",
                    "actionable_next_required_command": "actionable_next_required_command",
                    "actionable_next_proof_command": "actionable_next_proof_command",
                    "execution_learning_closure_actionable_next_required_command": "execution_learning_closure_actionable_next_required_command",
                    "execution_learning_closure_actionable_next_proof_command": "execution_learning_closure_actionable_next_proof_command",
                    "learning_closure_ready": "learning_closure_ready",
                    "blocks_completion_claim": "learning_closure_blocks_completion_claim",
                }.items():
                    if handoff.get(nested_key) != metadata.get(flat_key):
                        raise SystemExit(f"execution_learning_closure_packet handoff field {nested_key} diverged from {flat_key}: {metadata}")
                target = handoff.get("target") or {}
                for nested_key, flat_key in {
                    "run_id": "target_run_id",
                    "tool_name": "target_tool_name",
                    "toolset": "target_toolset",
                    "risk": "target_risk",
                    "ok": "target_ok",
                    "approved": "target_approved",
                    "approval_id": "target_approval_id",
                }.items():
                    if target.get(nested_key) != metadata.get(flat_key):
                        raise SystemExit(f"execution_learning_closure_packet target handoff field {nested_key} diverged: {metadata}")
                for key in ["review_only", "draft_only", "loads_without_execution"]:
                    if handoff.get(key) is not True or metadata.get(f"execution_learning_closure_{key}") is not True:
                        raise SystemExit(f"execution_learning_closure_packet handoff missed true {key}: {metadata}")
                for key in [
                    "authorizes_execution",
                    "authorizes_completion_claim",
                    "approval_granted",
                    "calls_model",
                    "executes_tools",
                    "writes_files",
                    "reads_personal_data",
                    "external_side_effect",
                    "controls_computer",
                    "queues_approval",
                ]:
                    if handoff.get(key) is not False:
                        raise SystemExit(f"execution_learning_closure_packet handoff should keep {key}=False: {metadata}")
                if (
                    metadata.get("execution_learning_closure_authorizes_execution") is not False
                    or metadata.get("execution_learning_closure_authorizes_completion_claim") is not False
                    or metadata.get("execution_learning_closure_approval_granted") is not False
                ):
                    raise SystemExit(f"execution_learning_closure_packet flat authority fields should be false: {metadata}")
                if (
                    metadata.get("executes_tools")
                    or metadata.get("queues_approval")
                    or metadata.get("approves_request")
                    or metadata.get("dismisses_request")
                    or metadata.get("reads_private_data")
                    or metadata.get("writes_files")
                    or metadata.get("writes_notes")
                    or metadata.get("writes_memory")
                    or metadata.get("controls_computer")
                ):
                    raise SystemExit("execution_learning_closure_packet should remain read-only.")

        bad_limit = recent_tool_runs({"limit": "not-a-number"})
        if bad_limit.metadata.get("limit") != 20 or bad_limit.metadata.get("writes_files"):
            raise SystemExit("recent_tool_runs should sanitize bad limits and remain read-only.")
        for malformed_counter in ("not-a-number", True, float("inf"), None):
            if _metadata_int(malformed_counter) != 0:
                raise SystemExit(f"audit metadata counters should default malformed values to zero: {malformed_counter!r}")
        if _metadata_int("7") != 7:
            raise SystemExit("audit metadata counters should preserve valid numeric strings.")

        huge_limit = recent_tool_runs({"limit": 999999})
        if huge_limit.metadata.get("limit") != 200:
            raise SystemExit("recent_tool_runs should clamp huge limits.")

        low_limit = recent_tool_runs({"limit": -10})
        if low_limit.metadata.get("limit") != 1:
            raise SystemExit("recent_tool_runs should clamp low limits.")
        for key in [
            "calls_model",
            "executes_tools",
            "queues_approval",
            "approves_request",
            "dismisses_request",
            "reads_private_data",
            "writes_files",
            "writes_notes",
            "writes_memory",
            "controls_computer",
            "external_side_effect",
            "requires_approval",
        ]:
            if low_limit.metadata.get(key) is not False:
                raise SystemExit(f"recent_tool_runs unsafe metadata {key}: {low_limit.metadata}")

        bool_limit = recent_tool_runs({"limit": True})
        if bool_limit.metadata.get("limit") != 20 or bool_limit.metadata.get("writes_files"):
            raise SystemExit(f"recent_tool_runs should treat boolean limits as malformed defaults: {bool_limit.metadata}")
        for key in [
            "calls_model",
            "executes_tools",
            "queues_approval",
            "approves_request",
            "dismisses_request",
            "reads_private_data",
            "writes_files",
            "writes_notes",
            "writes_memory",
            "controls_computer",
            "external_side_effect",
            "requires_approval",
        ]:
            if bool_limit.metadata.get(key) is not False:
                raise SystemExit(f"recent_tool_runs boolean limit unsafe metadata {key}: {bool_limit.metadata}")

        with TemporaryDirectory(prefix="jarvis-audit-recovery-receipt-") as recovery_temp:
            recovery_runtime = make_temp_runtime(Path(recovery_temp))
            _, recovery_verification_receipt, *_ = make_audit_tools(recovery_runtime.store)
            recovery_run_id = recovery_runtime.store.log_tool_run(
                recovery_runtime.session_id,
                "checkpoint_recovery_execute",
                "LOCAL_SAFE",
                False,
                False,
                "Execution held. A recovery step must be reviewed before Jarvis records it as applied.",
                metadata={
                    "checkpoint_recovery_execute_handoff_ready": True,
                    "checkpoint_recovery_execute_handoff_state": "HELD_BEFORE_REVIEW",
                    "checkpoint_recovery_execute_handoff_reviewed": False,
                    "checkpoint_recovery_execute_handoff_missing_fields": ["reviewed=true", "approved_step", "verification"],
                    "checkpoint_recovery_execute_handoff_missing_field_count": 3,
                    "checkpoint_recovery_execute_handoff_normal_followthrough_allowed": False,
                    "checkpoint_recovery_execute_handoff_recovery_followthrough_gate_state": "blocked_missing_reviewed_step",
                    "checkpoint_recovery_execute_handoff_recovery_closure_required_evidence": [
                        "reviewed recovery step",
                        "verification evidence",
                        "approval boundary",
                    ],
                    "checkpoint_recovery_execute_handoff_recovery_closure_required_evidence_count": 3,
                    "checkpoint_recovery_execute_handoff_recovery_closure_approval_boundary": "risky recovery steps require explicit approval before execution",
                    "checkpoint_recovery_execute_handoff_next_safe_command": "checkpoint recovery execute reviewed=true step=<reviewed local-safe step> verification=<test or evidence>",
                    "checkpoint_recovery_execute_handoff_risky_recovery_signal_count": 0,
                    "checkpoint_recovery_execute_handoff_approval_reference_provided": False,
                    "checkpoint_recovery_execute_handoff_recovery_step_approval_required_before_recovery": False,
                    "checkpoint_recovery_execute_handoff_recovery_step_approval_proof_queue": [],
                    "checkpoint_recovery_execute_handoff_recovery_step_approval_proof_queue_count": 0,
                    "checkpoint_recovery_execute_handoff_recovery_step_approval_boundary_token_sha256": "b" * 64,
                    "checkpoint_recovery_execute_handoff_recovery_step_approval_boundary_token_present": True,
                    "checkpoint_recovery_execute_handoff_token_sha256": "a" * 64,
                    "checkpoint_recovery_execute_handoff_token_present": True,
                    "checkpoint_recovery_execute_handoff_contains_raw_step": False,
                    "checkpoint_recovery_execute_handoff_contains_raw_verification": False,
                },
            )
            recovery_receipt = recovery_verification_receipt({"run_id": recovery_run_id})
            stale_recovery_run_id = recovery_runtime.store.log_tool_run(
                recovery_runtime.session_id,
                "checkpoint_recovery_execute",
                "LOCAL_SAFE",
                False,
                False,
                "Execution held. A recovery step must be reviewed before Jarvis records it as applied.",
                metadata={
                    "checkpoint_recovery_execute_handoff_ready": True,
                    "checkpoint_recovery_execute_handoff_state": "HELD_BEFORE_REVIEW",
                    "checkpoint_recovery_execute_handoff_reviewed": False,
                    "checkpoint_recovery_execute_handoff_missing_fields": ["reviewed=true", "approved_step", "verification"],
                    "checkpoint_recovery_execute_handoff_missing_field_count": 3,
                    "checkpoint_recovery_execute_handoff_normal_followthrough_allowed": False,
                    "checkpoint_recovery_execute_handoff_recovery_followthrough_gate_state": "blocked_missing_reviewed_step",
                    "checkpoint_recovery_execute_handoff_recovery_closure_required_evidence": [
                        "reviewed recovery step",
                        "verification evidence",
                        "approval boundary",
                    ],
                    "checkpoint_recovery_execute_handoff_recovery_closure_required_evidence_count": 3,
                    "checkpoint_recovery_execute_handoff_recovery_closure_approval_boundary": "risky recovery steps require explicit approval before execution",
                    "checkpoint_recovery_execute_handoff_next_safe_command": "checkpoint recovery execute reviewed=true step=<reviewed local-safe step> verification=<test or evidence>",
                    "checkpoint_recovery_execute_handoff_risky_recovery_signal_count": 0,
                    "checkpoint_recovery_execute_handoff_approval_reference_provided": False,
                    "checkpoint_recovery_execute_handoff_recovery_step_approval_required_before_recovery": False,
                    "checkpoint_recovery_execute_handoff_recovery_step_approval_proof_queue": [],
                    "checkpoint_recovery_execute_handoff_recovery_step_approval_proof_queue_count": 0,
                    "checkpoint_recovery_execute_handoff_recovery_step_approval_boundary_token_present": True,
                    "checkpoint_recovery_execute_handoff_contains_raw_step": False,
                    "checkpoint_recovery_execute_handoff_contains_raw_verification": False,
                },
            )
            stale_recovery_receipt = recovery_verification_receipt({"run_id": stale_recovery_run_id})
            stale_boundary_recovery_run_id = recovery_runtime.store.log_tool_run(
                recovery_runtime.session_id,
                "checkpoint_recovery_execute",
                "LOCAL_SAFE",
                False,
                False,
                "Execution held. A recovery step must be reviewed before Jarvis records it as applied.",
                metadata={
                    "checkpoint_recovery_execute_handoff_ready": True,
                    "checkpoint_recovery_execute_handoff_state": "HELD_BEFORE_REVIEW",
                    "checkpoint_recovery_execute_handoff_reviewed": False,
                    "checkpoint_recovery_execute_handoff_missing_fields": ["reviewed=true", "approved_step", "verification"],
                    "checkpoint_recovery_execute_handoff_missing_field_count": 3,
                    "checkpoint_recovery_execute_handoff_normal_followthrough_allowed": False,
                    "checkpoint_recovery_execute_handoff_recovery_followthrough_gate_state": "blocked_missing_reviewed_step",
                    "checkpoint_recovery_execute_handoff_recovery_closure_required_evidence": [
                        "reviewed recovery step",
                        "verification evidence",
                        "approval boundary",
                    ],
                    "checkpoint_recovery_execute_handoff_recovery_closure_required_evidence_count": 3,
                    "checkpoint_recovery_execute_handoff_recovery_closure_approval_boundary": "risky recovery steps require explicit approval before execution",
                    "checkpoint_recovery_execute_handoff_next_safe_command": "checkpoint recovery execute reviewed=true step=<reviewed local-safe step> verification=<test or evidence>",
                    "checkpoint_recovery_execute_handoff_risky_recovery_signal_count": 0,
                    "checkpoint_recovery_execute_handoff_approval_reference_provided": False,
                    "checkpoint_recovery_execute_handoff_recovery_step_approval_required_before_recovery": False,
                    "checkpoint_recovery_execute_handoff_recovery_step_approval_proof_queue": [],
                    "checkpoint_recovery_execute_handoff_recovery_step_approval_proof_queue_count": 0,
                    "checkpoint_recovery_execute_handoff_recovery_step_approval_boundary_token_sha256": "not-a-sha256-token",
                    "checkpoint_recovery_execute_handoff_recovery_step_approval_boundary_token_present": True,
                    "checkpoint_recovery_execute_handoff_token_sha256": "a" * 64,
                    "checkpoint_recovery_execute_handoff_token_present": True,
                    "checkpoint_recovery_execute_handoff_contains_raw_step": False,
                    "checkpoint_recovery_execute_handoff_contains_raw_verification": False,
                },
            )
            stale_boundary_recovery_receipt = recovery_verification_receipt({"run_id": stale_boundary_recovery_run_id})
        if not recovery_receipt.ok or recovery_receipt.metadata.get("verdict") != "FAILED_OR_BLOCKED":
            raise SystemExit(f"recovery verification receipt should inspect held recovery run: {recovery_receipt}")
        recovery_handoff = recovery_receipt.metadata.get("checkpoint_recovery_execute_handoff") or {}
        receipt_recovery_handoff = (
            recovery_receipt.metadata.get("verification_receipt_handoff", {}).get("checkpoint_recovery_execute_handoff")
            or {}
        )
        if recovery_handoff or receipt_recovery_handoff:
            raise SystemExit(
                f"recovery verification receipt should suppress flat-token-only handoff: {recovery_receipt.metadata}"
            )
        if recovery_receipt.metadata.get("verification_receipt_has_checkpoint_recovery_execute_handoff") is not False:
            raise SystemExit(
                f"recovery verification receipt should mark flat-token-only handoff absent: {recovery_receipt.metadata}"
            )
        if recovery_receipt.metadata.get("checkpoint_recovery_execute_handoff_invalid_reason"):
            raise SystemExit(
                f"recovery verification receipt should leave cleanly absent flat-token-only reason blank: {recovery_receipt.metadata}"
            )
        assert_no_future_authority(recovery_receipt.metadata, "recovery verification receipt")
        if stale_recovery_receipt.metadata.get("checkpoint_recovery_execute_handoff"):
            raise SystemExit(
                f"stale recovery verification receipt should suppress tokenless handoff: {stale_recovery_receipt.metadata}"
            )
        if stale_recovery_receipt.metadata.get("verification_receipt_has_checkpoint_recovery_execute_handoff") is not False:
            raise SystemExit(
                f"stale recovery verification receipt should mark handoff absent: {stale_recovery_receipt.metadata}"
            )
        if stale_recovery_receipt.metadata.get("checkpoint_recovery_execute_handoff_invalid_reason") != "invalid_or_stale_checkpoint_recovery_execute_handoff":
            raise SystemExit(
                f"stale recovery verification receipt should expose invalid handoff reason: {stale_recovery_receipt.metadata}"
            )
        if stale_boundary_recovery_receipt.metadata.get("checkpoint_recovery_execute_handoff"):
            raise SystemExit(
                f"stale boundary-token recovery receipt should suppress handoff: {stale_boundary_recovery_receipt.metadata}"
            )
        if (
            stale_boundary_recovery_receipt.metadata.get(
                "verification_receipt_has_checkpoint_recovery_execute_handoff"
            )
            is not False
        ):
            raise SystemExit(
                f"stale boundary-token recovery receipt should mark handoff absent: {stale_boundary_recovery_receipt.metadata}"
            )
        if stale_boundary_recovery_receipt.metadata.get("checkpoint_recovery_execute_handoff_invalid_reason") != "invalid_or_stale_checkpoint_recovery_execute_handoff":
            raise SystemExit(
                f"stale boundary-token recovery receipt should expose invalid handoff reason: {stale_boundary_recovery_receipt.metadata}"
            )

        missing_receipt = verification_receipt({"run_id": 99999})
        if missing_receipt.ok or missing_receipt.metadata.get("verdict") != "MISSING_RUN":
            raise SystemExit("verification_receipt should fail cleanly for unknown run ids.")
        missing_recovery = execution_recovery_packet({"run_id": 99999})
        missing_learning = after_action_learning_packet({"run_id": 99999})
        missing_learning_closure = execution_learning_closure_packet({"run_id": 99999})
        missing_trace_message = runtime_trace_receipt({"message_id": 99999})
        missing_lookup_cases = [
            (
                missing_receipt,
                "missing verification receipt run",
                ["recent tool runs", "verification receipt latest"],
            ),
            (
                missing_recovery,
                "missing execution recovery run",
                ["recent tool runs", "execution recovery"],
            ),
            (
                missing_learning,
                "missing after-action learning run",
                ["recent tool runs", "after action learning packet"],
            ),
            (
                missing_learning_closure,
                "missing execution learning closure run",
                ["recent tool runs", "execution learning closure"],
            ),
            (
                missing_trace_message,
                "missing runtime trace message",
                ["runtime trace receipt"],
            ),
        ]
        for missing_result, label, expected_commands in missing_lookup_cases:
            if missing_result.metadata.get("verdict") not in {"MISSING_RUN", "MISSING_MESSAGE"}:
                raise SystemExit(f"{label} missed stable missing-record verdict: {missing_result.metadata}")
            assert_audit_lookup_recovery(missing_result, label, expected_commands)

        bad_receipt = verification_receipt({"run_id": "run-abc"})
        if bad_receipt.ok or bad_receipt.metadata.get("verdict") != "INVALID_ID":
            raise SystemExit(f"verification_receipt should reject malformed run ids: {bad_receipt.metadata}")
        if bad_receipt.metadata.get("raw_run_id") != "run-abc":
            raise SystemExit(f"verification_receipt should preserve bounded raw run id: {bad_receipt.metadata}")

        zero_receipt = verification_receipt({"run_id": 0})
        if zero_receipt.ok or zero_receipt.metadata.get("verdict") != "INVALID_ID" or zero_receipt.metadata.get("raw_run_id") != "0":
            raise SystemExit(f"verification_receipt should reject non-positive run ids before lookup: {zero_receipt.metadata}")
        if "positive number" not in zero_receipt.output:
            raise SystemExit(f"verification_receipt should explain positive run id requirement: {zero_receipt.output}")

        long_bad_receipt = verification_receipt({"run_id": "r" * 200})
        if long_bad_receipt.ok or long_bad_receipt.metadata.get("raw_run_id") != ("r" * 79 + "…"):
            raise SystemExit(f"verification_receipt should bound raw run id: {long_bad_receipt.metadata}")
        path_bad_receipt = verification_receipt({"run_id": "/\x55sers/example/private/audit-run-id"})
        if path_bad_receipt.ok or path_bad_receipt.metadata.get("raw_run_id") != "<local-path>":
            raise SystemExit(f"verification_receipt should redact path-shaped bad run ids: {path_bad_receipt.metadata}")
        var_bad_receipt = verification_receipt({"run_id": "/var/folders/zc/audit-run-id"})
        if var_bad_receipt.ok or var_bad_receipt.metadata.get("raw_run_id") != "<local-path>":
            raise SystemExit(f"verification_receipt should redact macOS temp-root bad run ids: {var_bad_receipt.metadata}")

        no_expectation = verification_receipt({"run_id": "latest"})
        if no_expectation.metadata.get("verdict") not in {"PASS_WITH_AUDIT_EVIDENCE", "FAILED_OR_BLOCKED", "APPROVAL_EVIDENCE_MISSING"}:
            raise SystemExit(f"verification_receipt missed no-expectation verdict: {no_expectation.metadata}")

        direct_trace = runtime_trace_receipt({})
        if not direct_trace.ok or direct_trace.metadata.get("found") is not True:
            raise SystemExit(f"runtime_trace_receipt direct call missed latest trace: {direct_trace.metadata}")
        trace_rows = [
            row
            for row in runtime.store.recent_messages(limit=100)
            if row["role"] == "assistant" and '"runtime_trace"' in row["metadata"]
        ]
        if not trace_rows:
            raise SystemExit("Expected at least one assistant runtime trace row.")
        blocked_trace_id = None
        for row in trace_rows:
            trace_metadata = json.loads(row["metadata"] or "{}").get("runtime_trace", {})
            if trace_metadata.get("request") == "run command python3 --version":
                blocked_trace_id = int(row["id"])
                break
        if blocked_trace_id is None:
            raise SystemExit("Expected blocked shell command runtime trace row.")
        blocked_trace = runtime_trace_receipt({"message_id": blocked_trace_id})
        if blocked_trace.metadata.get("result_toolsets", {}).get("code") != 1:
            raise SystemExit(f"runtime trace receipt missed held result toolset metadata: {blocked_trace.metadata}")
        if "run_shell_command: failed/held, approval #1 [code, HIGH_RISK]" not in blocked_trace.output:
            raise SystemExit(f"runtime trace receipt missed held result risk/toolset evidence: {blocked_trace.output}")
        older_trace_id = int(trace_rows[0]["id"])
        exact_trace_small_window = runtime_trace_receipt({"message_id": older_trace_id, "limit": 1})
        if not exact_trace_small_window.ok or exact_trace_small_window.metadata.get("message_id") != older_trace_id:
            raise SystemExit(f"runtime_trace_receipt should find explicit message ids even with a tiny inspection window: {exact_trace_small_window.metadata}")
        if exact_trace_small_window.metadata.get("inspected_messages") != 1:
            raise SystemExit(f"runtime_trace_receipt should still report the bounded context window: {exact_trace_small_window.metadata}")
        if f"message id: #{older_trace_id}" not in exact_trace_small_window.output:
            raise SystemExit("runtime_trace_receipt exact lookup output missed the requested message id.")
        planned_trace = runtime.handle(f"runtime trace receipt {older_trace_id}")
        if not planned_trace.verified or planned_trace.tool_results[0].metadata.get("message_id") != older_trace_id:
            raise SystemExit(f"Planner should route numbered runtime trace receipts to exact message lookup: {planned_trace.tool_results[0].metadata}")
        user_message_id = int(next(row["id"] for row in runtime.store.recent_messages(limit=100) if row["role"] == "user"))
        user_trace = runtime_trace_receipt({"message_id": user_message_id})
        if user_trace.ok or user_trace.metadata.get("verdict") != "MISSING_TRACE":
            raise SystemExit(f"runtime_trace_receipt should reject user messages without trace metadata: {user_trace.metadata}")
        for expected in (
            "Send a Jarvis command first",
            "runtime trace receipt",
            "without an ID",
        ):
            if expected not in user_trace.output:
                raise SystemExit(
                    f"runtime_trace_receipt missing-trace output missed recovery text "
                    f"{expected!r}: {user_trace.output}"
                )
        if user_trace.metadata.get("next_command") != "send a Jarvis command":
            raise SystemExit(
                f"runtime_trace_receipt missing-trace result missed next command: "
                f"{user_trace.metadata}"
            )
        if user_trace.metadata.get("recovery_commands") != [
            "send a Jarvis command",
            "runtime trace receipt",
        ]:
            raise SystemExit(
                f"runtime_trace_receipt missing-trace result missed bounded recovery commands: "
                f"{user_trace.metadata}"
            )
        if user_trace.metadata.get("retry_requires_fresh_runtime_trace") is not True:
            raise SystemExit(
                f"runtime_trace_receipt missing-trace result should require a fresh trace: "
                f"{user_trace.metadata}"
            )
        assert_no_future_authority(user_trace.metadata, "runtime_trace_receipt missing trace")

        malformed_trace_id = runtime.store.log_message(
            runtime.session_id,
            "assistant",
            "malformed runtime trace fixture",
            {
                "runtime_trace": {
                    "session_id": runtime.session_id,
                    "route": "test",
                    "verified": "true",
                    "approval_required": "true",
                    "ran_tool_handlers": "true",
                    "approval_queue_before": "bad-before",
                    "approval_queue_after": "bad-after",
                    "approval_queue_delta": "bad-delta",
                    "approved_reruns": "bad-reruns",
                    "planner_metadata": {
                        "model_planner_attempted": "true",
                        "model_planner_used": "true",
                        "model_planner_fell_back": "true",
                        "authorizes_execution": "true",
                        "authorizes_completion_claim": "true",
                        "approval_granted": "true",
                    },
                    "tool_results": [],
                }
            },
        )
        malformed_trace = runtime_trace_receipt({"message_id": malformed_trace_id})
        if not malformed_trace.ok or malformed_trace.metadata.get("message_id") != malformed_trace_id:
            raise SystemExit(f"runtime_trace_receipt should handle explicit malformed trace metadata: {malformed_trace.metadata}")
        for key in ("approval_queue_before", "approval_queue_after", "approval_queue_delta", "approved_reruns"):
            if malformed_trace.metadata.get(key) != 0:
                raise SystemExit(f"runtime_trace_receipt should default malformed {key} to zero: {malformed_trace.metadata}")
        for key in ("verified", "approval_required", "planner_model_planner_attempted"):
            if malformed_trace.metadata.get(key) is not False:
                raise SystemExit(f"runtime_trace_receipt should default malformed boolean {key} to false: {malformed_trace.metadata}")
        malformed_planner_metadata = malformed_trace.metadata.get("planner_metadata") or {}
        for key in (
            "model_planner_attempted",
            "model_planner_used",
            "model_planner_fell_back",
            "authorizes_execution",
            "authorizes_completion_claim",
            "approval_granted",
        ):
            if malformed_planner_metadata.get(key) is not False:
                raise SystemExit(f"runtime_trace_receipt should not trust malformed planner boolean {key}: {malformed_trace.metadata}")
        if malformed_trace.metadata.get("verdict") != "HELD_OR_FAILED":
            raise SystemExit(f"runtime_trace_receipt should not verify malformed boolean traces: {malformed_trace.metadata}")
        if "- verified: no" not in malformed_trace.output or "- model planner: not attempted" not in malformed_trace.output:
            raise SystemExit(f"runtime_trace_receipt should print malformed boolean trace as conservative false: {malformed_trace.output}")

        stale_checkpoint_trace_id = runtime.store.log_message(
            runtime.session_id,
            "assistant",
            "stale checkpoint recovery trace fixture",
            {
                "runtime_trace": {
                    "session_id": runtime.session_id,
                    "request": "verification receipt latest",
                    "route": "tools",
                    "goal": "Inspect latest verification evidence.",
                    "verified": True,
                    "approval_required": False,
                    "ran_tool_handlers": True,
                    "approval_queue_before": 0,
                    "approval_queue_after": 0,
                    "approval_queue_delta": 0,
                    "stages": [
                        {"stage": "verification", "status": "ok", "detail": "All planned actions completed."}
                    ],
                    "planned_actions": [],
                    "tool_results": [],
                    "risk_levels": ["READ_ONLY"],
                },
                "tool_results": [
                    {
                        "tool": "verification_receipt",
                        "metadata": {
                            "checkpoint_recovery_execute_handoff": {
                                "source": "checkpoint_recovery_execute",
                                "state": "HELD_BEFORE_REVIEW",
                            },
                            "checkpoint_recovery_execute_handoff_ready": True,
                            "checkpoint_recovery_execute_handoff_token_present": True,
                            "checkpoint_recovery_execute_handoff_recovery_step_approval_boundary_token_present": True,
                        },
                    }
                ],
            },
        )
        stale_checkpoint_trace = runtime_trace_receipt({"message_id": stale_checkpoint_trace_id})
        if not stale_checkpoint_trace.ok:
            raise SystemExit(f"runtime_trace_receipt should inspect stale checkpoint trace metadata: {stale_checkpoint_trace.metadata}")
        if stale_checkpoint_trace.metadata.get("checkpoint_recovery_execute_handoff"):
            raise SystemExit(f"runtime_trace_receipt should suppress invalid checkpoint recovery handoffs: {stale_checkpoint_trace.metadata}")
        if stale_checkpoint_trace.metadata.get("runtime_trace_receipt_has_checkpoint_recovery_execute_handoff") is not False:
            raise SystemExit(f"runtime_trace_receipt should report no valid checkpoint handoff: {stale_checkpoint_trace.metadata}")
        if stale_checkpoint_trace.metadata.get("checkpoint_recovery_execute_handoff_invalid_reason") != "invalid_or_stale_checkpoint_recovery_execute_handoff":
            raise SystemExit(f"runtime_trace_receipt should preserve stale checkpoint invalid reason: {stale_checkpoint_trace.metadata}")
        if (
            stale_checkpoint_trace.metadata.get(
                "runtime_trace_receipt_checkpoint_recovery_execute_handoff_invalid_reason"
            )
            != "invalid_or_stale_checkpoint_recovery_execute_handoff"
        ):
            raise SystemExit(f"runtime_trace_receipt missed flat runtime-trace invalid reason: {stale_checkpoint_trace.metadata}")
        stale_checkpoint_handoff = stale_checkpoint_trace.metadata.get("runtime_trace_receipt_handoff") or {}
        if stale_checkpoint_handoff.get("checkpoint_recovery_execute_handoff"):
            raise SystemExit(f"runtime_trace_receipt handoff should suppress invalid checkpoint handoffs: {stale_checkpoint_trace.metadata}")
        if stale_checkpoint_handoff.get("checkpoint_recovery_execute_handoff_invalid_reason") != "invalid_or_stale_checkpoint_recovery_execute_handoff":
            raise SystemExit(f"runtime_trace_receipt handoff missed invalid reason: {stale_checkpoint_trace.metadata}")
        if "invalid_or_stale_checkpoint_recovery_execute_handoff" not in stale_checkpoint_trace.output:
            raise SystemExit(f"runtime_trace_receipt output missed checkpoint invalid reason: {stale_checkpoint_trace.output}")

        planner_trace_id = runtime.store.log_message(
            runtime.session_id,
            "assistant",
            "model planner fallback trace fixture",
            {
                "runtime_trace": {
                    "session_id": runtime.session_id,
                    "request": "please frobnicate workspace",
                    "route": "chat",
                    "goal": "Respond conversationally.",
                    "needs_model": True,
                    "planner_notes": "This should route to a model planner once configured.",
                    "planner_metadata": {
                        "planner_type": "model_backed",
                        "model_planner_attempted": True,
                        "model_planner_state": "fallback",
                        "model_planner_used": False,
                        "model_planner_fell_back": True,
                        "model_planner_fallback_reason": "no_valid_model_actions",
                        "model_planner_fallback_detail": "missing tool near /\x55sers/example/private/planner plus /tmp/jarvis-plan",
                        "model_planner_recovery_hint": "Run `model routing status`, start Ollama if it is stopped, and run `ollama pull jarvis-v2-smoke-planner` outside Jarvis if the planner model is missing. Diagnostic: planner_model_unavailable.",
                        "model_planner_model": "jarvis-v2-smoke-planner",
                        "model_planner_timeout_seconds": 1.5,
                        "model_planner_action_count": 0,
                        "model_planner_ignored_unknown_tools": ["missing_secret_tool", "/private/tmp/jarvis-secret-tool"],
                        "authorizes_execution": False,
                        "authorizes_completion_claim": False,
                        "approval_granted": False,
                    },
                    "verified": True,
                    "approval_required": False,
                    "ran_tool_handlers": False,
                    "approval_queue_before": 0,
                    "approval_queue_after": 0,
                    "approval_queue_delta": 0,
                    "stages": [
                        {
                            "stage": "planning",
                            "status": "ok",
                            "detail": "Respond conversationally.",
                            "planner_metadata": {"model_planner_fallback_reason": "no_valid_model_actions"},
                        }
                    ],
                    "planned_actions": [],
                    "tool_results": [],
                    "risk_levels": [],
                }
            },
        )
        planner_trace = runtime_trace_receipt({"message_id": planner_trace_id})
        if not planner_trace.ok:
            raise SystemExit(f"runtime_trace_receipt should inspect planner fallback traces: {planner_trace.metadata}")
        for expected in [
            "Planner diagnostics",
            "model planner state: fallback",
            "model planner fallback: yes",
            "model planner fallback reason: no_valid_model_actions",
            "ignored model tools: missing_secret_tool, <local-path>",
            "model planner detail: missing tool near <local-path>",
            "model planner recovery: Run `model routing status`, start Ollama if it is stopped",
            "Diagnostic: planner_model_unavailable",
        ]:
            if expected not in planner_trace.output:
                raise SystemExit(f"runtime trace receipt missed planner diagnostic text {expected!r}: {planner_trace.output}")
        if any(fragment in planner_trace.output for fragment in ["/\x55sers/", "/private/", "/tmp/"]):
            raise SystemExit(f"runtime trace planner diagnostics leaked local paths: {planner_trace.output}")
        planner_metadata = planner_trace.metadata.get("planner_metadata") or {}
        if (
            planner_trace.metadata.get("planner_model_planner_attempted") is not True
            or planner_trace.metadata.get("planner_model_planner_state") != "fallback"
            or planner_trace.metadata.get("planner_model_planner_fallback_reason") != "no_valid_model_actions"
            or planner_trace.metadata.get("planner_model_planner_ignored_unknown_tools") != ["missing_secret_tool", "<local-path>"]
            or planner_metadata.get("model_planner_fallback_detail") != "missing tool near <local-path>"
            or "model routing status" not in str(planner_metadata.get("model_planner_recovery_hint") or "")
            or planner_trace.metadata.get("planner_model_planner_recovery_hint") != planner_metadata.get("model_planner_recovery_hint")
        ):
            raise SystemExit(f"runtime trace receipt missed flat/nested planner metadata: {planner_trace.metadata}")
        if any(fragment in str(planner_trace.metadata) for fragment in ["/\x55sers/", "/private/", "/tmp/"]):
            raise SystemExit(f"runtime trace planner metadata leaked local paths: {planner_trace.metadata}")
        planner_handoff = planner_trace.metadata.get("runtime_trace_receipt_handoff") or {}
        if planner_handoff.get("planner_metadata") != planner_metadata:
            raise SystemExit(f"runtime trace planner handoff should mirror planner metadata: {planner_trace.metadata}")
        for key in ["authorizes_execution", "authorizes_completion_claim", "approval_granted"]:
            if planner_metadata.get(key) is not False or planner_handoff.get("planner_metadata", {}).get(key) is not False:
                raise SystemExit(f"runtime trace planner diagnostics should grant no authority: {planner_trace.metadata}")

        bad_trace = runtime_trace_receipt({"message_id": "trace-abc"})
        if bad_trace.ok or bad_trace.metadata.get("verdict") != "INVALID_ID":
            raise SystemExit(f"runtime_trace_receipt should reject malformed message ids: {bad_trace.metadata}")
        if bad_trace.metadata.get("raw_message_id") != "trace-abc":
            raise SystemExit(f"runtime_trace_receipt should preserve bounded raw message id: {bad_trace.metadata}")
        path_bad_trace = runtime_trace_receipt({"message_id": "/private/tmp/jarvis-trace-id"})
        if path_bad_trace.ok or path_bad_trace.metadata.get("raw_message_id") != "<local-path>":
            raise SystemExit(f"runtime_trace_receipt should redact path-shaped bad message ids: {path_bad_trace.metadata}")
        tmp_bad_trace = runtime_trace_receipt({"message_id": "/tmp/jarvis-trace-id"})
        if tmp_bad_trace.ok or tmp_bad_trace.metadata.get("raw_message_id") != "<local-path>":
            raise SystemExit(f"runtime_trace_receipt should redact tmp-root bad message ids: {tmp_bad_trace.metadata}")

        zero_trace = runtime_trace_receipt({"message_id": 0})
        if zero_trace.ok or zero_trace.metadata.get("verdict") != "INVALID_ID" or zero_trace.metadata.get("raw_message_id") != "0":
            raise SystemExit(f"runtime_trace_receipt should reject non-positive message ids before lookup: {zero_trace.metadata}")
        if "positive number" not in zero_trace.output:
            raise SystemExit(f"runtime_trace_receipt should explain positive message id requirement: {zero_trace.output}")

        noisy_audit_run_id = runtime.store.log_tool_run(
            runtime.session_id,
            "noisy_audit_fixture",
            "LOCAL_SAFE",
            True,
            False,
            "opened /\x55sers/example/Desktop/Claude code/AI agents/jarvis-v2/private.txt then checked /private/tmp/jarvis-secret.log and /var/folders/zc/jarvis-secret.log plus /tmp/jarvis-audit-secret.log",
            metadata={"toolset": "test"},
        )
        noisy_recent = recent_tool_runs({"limit": 1})
        noisy_receipt = verification_receipt({"run_id": noisy_audit_run_id})
        for label, output in {"recent": noisy_recent.output, "receipt": noisy_receipt.output}.items():
            if any(fragment in output for fragment in ["/\x55sers/", "/private/", "/var/folders/", "/tmp/"]):
                raise SystemExit(f"audit {label} output leaked a local path: {output}")
            if "<local-path>" not in output:
                raise SystemExit(f"audit {label} output should show local path redaction: {output}")

        direct_health = execution_health_report({"limit": 999999})
        if not direct_health.ok or direct_health.metadata.get("inspected_runs", 0) < 1:
            raise SystemExit(f"execution_health_report direct call missed audit rows: {direct_health.metadata}")
        if direct_health.metadata.get("limit") is not None:
            raise SystemExit("execution_health_report should not expose unsanitized raw limit metadata.")
        planned_health = runtime.handle("runtime health")
        if not planned_health.verified or planned_health.tool_results[0].tool_name != "execution_health_report":
            raise SystemExit(f"Planner should route runtime health to execution_health_report: {planned_health.response}")

        with TemporaryDirectory(prefix="jarvis-audit-meta-target-") as meta_temp:
            meta_runtime = make_temp_runtime(Path(meta_temp))
            meta_recent_tool_runs, _meta_verification_receipt, _meta_runtime_trace_receipt, _meta_execution_audit_gate, _meta_execution_recovery_packet, _meta_after_action_learning_packet, meta_execution_health_report, _meta_recovery_closure_checklist, meta_execution_learning_closure_packet = make_audit_tools(meta_runtime.store)
            action_run_id = meta_runtime.store.log_tool_run(
                session_id="meta-target",
                tool_name="calculate",
                risk="READ_ONLY",
                ok=True,
                approved=False,
                output="4",
                metadata={"route": "tools"},
            )
            meta_runtime.store.log_tool_run(
                session_id="meta-target",
                tool_name="completion_next_proof_packet",
                risk="READ_ONLY",
                ok=True,
                approved=False,
                output="Jarvis completion next proof packet",
                metadata={"next_proof_command": f"execution learning closure {action_run_id}"},
            )
            meta_runtime.store.log_tool_run(
                session_id="meta-target",
                tool_name="harness_readiness_digest",
                risk="READ_ONLY",
                ok=True,
                approved=False,
                output="Jarvis harness readiness digest",
                metadata={"source_tool": "completion_next_proof_packet"},
            )
            meta_health = meta_execution_health_report({"limit": 20})
            if meta_health.metadata.get("learning_target_run_id") != action_run_id:
                raise SystemExit(f"execution_health_report should ignore completion/readiness proof packets as learning targets: {meta_health.metadata}")
            if meta_health.metadata.get("learning_target_tool") != "calculate":
                raise SystemExit(f"execution_health_report selected the wrong non-meta learning target: {meta_health.metadata}")
            if meta_health.metadata.get("action_runs") != 1 or meta_health.metadata.get("meta_runs", 0) < 2:
                raise SystemExit(f"execution_health_report should classify completion/readiness packets as meta runs: {meta_health.metadata}")
            meta_closure = meta_execution_learning_closure_packet({})
            if meta_closure.metadata.get("target_run_id") != action_run_id or meta_closure.metadata.get("target_tool_name") != "calculate":
                raise SystemExit(f"execution_learning_closure_packet should inherit the stable non-meta target: {meta_closure.metadata}")
            latest_target = meta_recent_tool_runs({}).metadata
            if latest_target.get("count", 0) < 3:
                raise SystemExit(f"recent_tool_runs sanity check missed meta target rows: {latest_target}")

        with TemporaryDirectory(prefix="jarvis-audit-repeated-") as repeated_temp:
            repeated_runtime = make_temp_runtime(Path(repeated_temp))
            repeated_audit_tools = make_audit_tools(repeated_runtime.store)
            repeated_execution_health_report = repeated_audit_tools[6]
            _repeated_recovery_closure_checklist = repeated_audit_tools[7]
            first_repeated_block = repeated_runtime.handle("run command python3 --version")
            second_repeated_block = repeated_runtime.handle("run command python3 --version")
            if first_repeated_block.verified or second_repeated_block.verified:
                raise SystemExit("Expected repeated shell commands to stay approval-gated for repeated failure health test.")
            repeated_health = repeated_execution_health_report({"limit": 80})
        if "Failure promotion queue:" not in repeated_health.output:
            raise SystemExit(f"execution_health_report missed failure promotion section: {repeated_health.output}")
        repeated_metadata = repeated_health.metadata
        if repeated_metadata.get("active_approval_holds", 0) < 2:
            raise SystemExit(f"execution_health_report missed repeated active approval holds: {repeated_metadata}")
        if repeated_metadata.get("repeated_failure_count") != 0:
            raise SystemExit(f"approval holds must not create repeated failure promotion work: {repeated_metadata}")
        if repeated_metadata.get("failure_promotion_queue") != []:
            raise SystemExit(f"approval holds must not enter the failure promotion queue: {repeated_metadata}")

        direct_gate = execution_audit_gate({"limit": 30})
        if not direct_gate.ok or direct_gate.metadata.get("inspected_runs", 0) < 1:
            raise SystemExit(f"execution_audit_gate direct call missed audit metadata: {direct_gate.metadata}")
        for expected in [
            "Jarvis execution audit gate",
            "Verdict:",
            "Inspected runs:",
            "execution recovery packet 2",
            "verification receipt 2",
            "Execution proof queue:",
            "next required command:",
            "proof queue count:",
            "proof queue:",
            "Boundary:",
            "does not call models",
        ]:
            if expected not in direct_gate.output:
                raise SystemExit(f"execution_audit_gate output missing expected text: {expected}")
        if "Execution proof queue:\n- next proof command:" in direct_gate.output:
            raise SystemExit("execution_audit_gate should render execution queue head as next required command.")
        if direct_gate.metadata.get("next_recovery_command") != "execution recovery packet 2":
            raise SystemExit(f"execution_audit_gate missed next recovery metadata: {direct_gate.metadata}")
        gate_queue = direct_gate.metadata.get("audit_proof_queue", [])
        if not gate_queue or direct_gate.metadata.get("proof_queue") != gate_queue:
            raise SystemExit(f"execution_audit_gate missed proof queue metadata: {direct_gate.metadata}")
        if direct_gate.metadata.get("audit_proof_queue_count") != len(gate_queue) or direct_gate.metadata.get("proof_queue_count") != len(gate_queue):
            raise SystemExit(f"execution_audit_gate missed proof queue count metadata: {direct_gate.metadata}")
        if direct_gate.metadata.get("audit_next_proof_command") != gate_queue[0] or direct_gate.metadata.get("next_proof_command") != gate_queue[0]:
            raise SystemExit(f"execution_audit_gate missed next proof command metadata: {direct_gate.metadata}")
        if direct_gate.metadata.get("audit_next_required_command") != gate_queue[0] or direct_gate.metadata.get("next_required_command") != gate_queue[0]:
            raise SystemExit(f"execution_audit_gate missed next required command metadata: {direct_gate.metadata}")
        for expected_command in [
            "verification receipt 2",
            "execution recovery packet 2",
            "execution learning closure 2",
            "after-action learning packet 2",
            "execution health report",
            "execution audit gate",
        ]:
            if expected_command not in gate_queue:
                raise SystemExit(f"execution_audit_gate missed proof queue command {expected_command}: {direct_gate.metadata}")
        gate_handoff = direct_gate.metadata.get("execution_audit_handoff") or {}
        if gate_handoff.get("source") != "execution_audit_gate":
            raise SystemExit(f"execution_audit_gate missed structured handoff: {direct_gate.metadata}")
        for nested_key, flat_key in {
            "verdict": "verdict",
            "review_required": "review_required",
            "safe_to_trust_recent_execution": "safe_to_trust_recent_execution",
            "inspected_runs": "inspected_runs",
            "failed_or_blocked_runs": "failed_or_blocked_runs",
            "risk_gated_runs": "risk_gated_runs",
            "risky_unapproved_successes": "risky_unapproved_successes",
            "risky_runs_missing_approval_ids": "risky_runs_missing_approval_ids",
            "verification_runs": "verification_runs",
            "newest_run_id": "newest_run_id",
            "newest_problem_run_id": "newest_problem_run_id",
            "problem_runs": "problem_runs",
            "problem_run_count": "problem_run_count",
            "next_recovery_command": "next_recovery_command",
            "audit_proof_queue": "audit_proof_queue",
            "audit_proof_queue_count": "audit_proof_queue_count",
            "audit_next_proof_command": "audit_next_proof_command",
            "proof_queue": "proof_queue",
            "proof_queue_count": "proof_queue_count",
            "next_proof_command": "next_proof_command",
        }.items():
            if gate_handoff.get(nested_key) != direct_gate.metadata.get(flat_key):
                raise SystemExit(f"execution_audit_gate handoff field {nested_key} diverged from {flat_key}: {direct_gate.metadata}")
        if gate_handoff.get("safe_to_trust_recent_execution") != (direct_gate.metadata.get("review_required") is False):
            raise SystemExit(f"execution_audit_gate handoff safe-to-trust flag diverged: {direct_gate.metadata}")
        if not gate_handoff.get("problem_runs") or gate_handoff["problem_runs"][0].get("run_id") != 2:
            raise SystemExit(f"execution_audit_gate handoff missed first problem run: {direct_gate.metadata}")
        for key in ["review_only", "draft_only", "loads_without_execution"]:
            if gate_handoff.get(key) is not True or direct_gate.metadata.get(f"execution_audit_{key}") is not True:
                raise SystemExit(f"execution_audit_gate handoff missed true {key}: {direct_gate.metadata}")
        for key in [
            "authorizes_execution",
            "authorizes_completion_claim",
            "approval_granted",
            "calls_model",
            "executes_tools",
            "writes_files",
            "reads_personal_data",
            "external_side_effect",
            "controls_computer",
            "queues_approval",
        ]:
            if gate_handoff.get(key) is not False:
                raise SystemExit(f"execution_audit_gate handoff should keep {key}=False: {direct_gate.metadata}")
        if (
            direct_gate.metadata.get("execution_audit_authorizes_execution") is not False
            or direct_gate.metadata.get("execution_audit_authorizes_completion_claim") is not False
            or direct_gate.metadata.get("execution_audit_approval_granted") is not False
        ):
            raise SystemExit(f"execution_audit_gate flat authority fields should be false: {direct_gate.metadata}")
        for key in [
            "calls_model",
            "executes_tools",
            "queues_approval",
            "approves_request",
            "dismisses_request",
            "reads_private_data",
            "writes_files",
            "writes_notes",
            "writes_memory",
            "controls_computer",
            "external_side_effect",
            "requires_approval",
        ]:
            if direct_gate.metadata.get(key) is not False:
                raise SystemExit(f"execution_audit_gate unsafe metadata {key}: {direct_gate.metadata}")

        direct_recovery = execution_recovery_packet({"run_id": 2})
        if not direct_recovery.ok or direct_recovery.metadata.get("run_id") != 2:
            raise SystemExit(f"execution_recovery_packet direct call missed run #2: {direct_recovery.metadata}")
        if direct_recovery.metadata.get("verdict") != "HELD_FOR_APPROVAL":
            raise SystemExit(f"execution_recovery_packet direct call missed held verdict: {direct_recovery.metadata}")
        for expected in [
            "verification receipt 2",
            "approval packet 1",
            "approval chain proof 1",
            "verification receipt <approved run id from approval chain proof 1>",
            "toolset: code",
            "Regression preview command:",
            "Next command queue",
            "Recovery closure checklist",
            "Retry readiness: blocked",
            "explicit stop times",
            "failure to test preview: tool run #2 run_shell_command produced HELD_FOR_APPROVAL",
            "turn the command, blocker, expected behavior, and verification target into a smoke test",
        ]:
            if expected not in direct_recovery.output:
                raise SystemExit(f"execution_recovery_packet direct output missing expected text: {expected}")
        if "failure to test preview" not in str(direct_recovery.metadata.get("failure_to_test_command", "")):
            raise SystemExit(f"execution_recovery_packet missed failure-to-test metadata: {direct_recovery.metadata}")
        if direct_recovery.metadata.get("toolset") != "code":
            raise SystemExit(f"execution_recovery_packet direct call missed toolset metadata: {direct_recovery.metadata}")
        for expected_command in [
            "verification receipt 2",
            "approval readiness 1",
            "approval packet 1",
            "approval chain proof 1",
            "verification receipt <approved run id from approval chain proof 1>",
            "execution learning closure 2",
            "after-action learning packet 2",
            "execution health report",
        ]:
            if expected_command not in direct_recovery.metadata.get("next_commands", []):
                raise SystemExit(f"execution_recovery_packet direct call missed queued next command {expected_command}: {direct_recovery.metadata}")
        if direct_recovery.metadata.get("approval_proof_chains", {}).get("1") != [
            "approval readiness 1",
            "approval packet 1",
            "approve approval 1",
            "approval chain proof 1",
            "verification receipt <approved run id from approval chain proof 1>",
        ]:
            raise SystemExit(f"execution_recovery_packet direct call missed approval proof-chain metadata: {direct_recovery.metadata}")
        if direct_recovery.metadata.get("next_command") != "verification receipt 2" or direct_recovery.metadata.get("next_command_count", 0) < 5:
            raise SystemExit(f"execution_recovery_packet direct call missed ordered next-command metadata: {direct_recovery.metadata}")
        if direct_recovery.metadata.get("recovery_closure_blocks_retry") is not True:
            raise SystemExit(f"execution_recovery_packet direct call missed recovery-closure retry block: {direct_recovery.metadata}")
        if direct_recovery.metadata.get("recovery_closure_next_required_command") != "verification receipt 2":
            raise SystemExit(f"execution_recovery_packet direct call missed first recovery-closure command: {direct_recovery.metadata}")
        direct_closure_commands = direct_recovery.metadata.get("recovery_closure_required_commands", [])
        if direct_recovery.metadata.get("recovery_closure_proof_queue") != direct_closure_commands:
            raise SystemExit(f"execution_recovery_packet direct call proof queue should mirror required commands: {direct_recovery.metadata}")
        if direct_recovery.metadata.get("recovery_closure_proof_queue_count") != len(direct_closure_commands):
            raise SystemExit(f"execution_recovery_packet direct call proof queue count diverged: {direct_recovery.metadata}")
        if direct_recovery.metadata.get("recovery_closure_next_proof_command") != "verification receipt 2":
            raise SystemExit(f"execution_recovery_packet direct call missed next recovery proof command: {direct_recovery.metadata}")
        for expected_command in [
            "verification receipt 2",
            "execution recovery packet 2",
            "execution learning closure 2",
            "after-action learning packet 2",
            "execution audit gate",
            "approval chain proof 1",
        ]:
            if expected_command not in direct_recovery.metadata.get("recovery_closure_required_commands", []):
                raise SystemExit(f"execution_recovery_packet direct call missed closure command {expected_command}: {direct_recovery.metadata}")
        if direct_recovery.metadata.get("operator_timeboxes_override_priority") is not True or direct_recovery.metadata.get("stop_times_override_priority") is not True:
            raise SystemExit(f"execution_recovery_packet direct call missed operator-limit metadata: {direct_recovery.metadata}")
        for key in [
            "calls_model",
            "executes_tools",
            "queues_approval",
            "approves_request",
            "dismisses_request",
            "reads_private_data",
            "writes_files",
            "writes_notes",
            "writes_memory",
            "controls_computer",
            "external_side_effect",
            "requires_approval",
        ]:
            if direct_recovery.metadata.get(key) is not False:
                raise SystemExit(f"execution_recovery_packet unsafe metadata {key}: {direct_recovery.metadata}")

        direct_learning = after_action_learning_packet({"run_id": 2})
        if not direct_learning.ok or direct_learning.metadata.get("run_id") != 2:
            raise SystemExit(f"after_action_learning_packet direct call missed run #2: {direct_learning.metadata}")
        if direct_learning.metadata.get("verdict") != "PROMOTE_FAILURE_TO_REVIEW":
            raise SystemExit(f"after_action_learning_packet should promote failed run #2: {direct_learning.metadata}")
        for expected in [
            "Jarvis after-action learning packet",
            "regression-test candidate from failed or blocked execution",
            "verification receipt 2",
            "execution recovery packet 2",
            "approval chain proof 1",
            "verification receipt <approved run id from approval chain proof 1>",
            "toolset: code",
            "failure to test preview: tool run #2 run_shell_command",
            "Next command queue",
            "Optional local-safe follow-up after human review",
            "explicit stop times",
        ]:
            if expected not in direct_learning.output:
                raise SystemExit(f"after_action_learning_packet direct output missing expected text: {expected}")
        if "failure to test preview" not in str(direct_learning.metadata.get("regression_command", "")):
            raise SystemExit(f"after_action_learning_packet missed regression metadata: {direct_learning.metadata}")
        if direct_learning.metadata.get("toolset") != "code":
            raise SystemExit(f"after_action_learning_packet direct call missed toolset metadata: {direct_learning.metadata}")
        for expected_command in [
            "verification receipt 2",
            "execution recovery packet 2",
            "approval readiness 1",
            "approval packet 1",
            "approval chain proof 1",
            "verification receipt <approved run id from approval chain proof 1>",
            "learning review",
        ]:
            if expected_command not in direct_learning.metadata.get("next_commands", []):
                raise SystemExit(f"after_action_learning_packet direct call missed queued next command {expected_command}: {direct_learning.metadata}")
        if direct_learning.metadata.get("approval_proof_chains", {}).get("1") != [
            "approval readiness 1",
            "approval packet 1",
            "approve approval 1",
            "approval chain proof 1",
            "verification receipt <approved run id from approval chain proof 1>",
        ]:
            raise SystemExit(f"after_action_learning_packet direct call missed approval proof-chain metadata: {direct_learning.metadata}")
        if direct_learning.metadata.get("next_command") != "verification receipt 2" or direct_learning.metadata.get("next_command_count", 0) < 5:
            raise SystemExit(f"after_action_learning_packet direct call missed ordered next-command metadata: {direct_learning.metadata}")
        direct_learning_handoff = direct_learning.metadata.get("after_action_learning_handoff") or {}
        if direct_learning_handoff.get("source") != "after_action_learning_packet":
            raise SystemExit(f"after_action_learning_packet direct call missed handoff source: {direct_learning.metadata}")
        if direct_learning_handoff.get("target", {}).get("run_id") != 2 or direct_learning_handoff.get("target", {}).get("tool_name") != "run_shell_command":
            raise SystemExit(f"after_action_learning_packet direct call missed handoff target: {direct_learning.metadata}")
        if direct_learning_handoff.get("approval", {}).get("proof_chains") != direct_learning.metadata.get("approval_proof_chains"):
            raise SystemExit(f"after_action_learning_packet direct call missed handoff approval chains: {direct_learning.metadata}")
        if direct_learning_handoff.get("proof_queue") != direct_learning.metadata.get("proof_queue"):
            raise SystemExit(f"after_action_learning_packet direct call missed handoff proof queue: {direct_learning.metadata}")
        for key in ["review_only", "draft_only", "loads_without_execution"]:
            if direct_learning_handoff.get(key) is not True or direct_learning.metadata.get(f"after_action_learning_{key}") is not True:
                raise SystemExit(f"after_action_learning_packet direct call missed handoff {key}: {direct_learning.metadata}")
        for key in [
            "authorizes_execution",
            "authorizes_completion_claim",
            "approval_granted",
            "calls_model",
            "executes_tools",
            "writes_files",
            "reads_personal_data",
            "external_side_effect",
            "controls_computer",
            "queues_approval",
        ]:
            if direct_learning_handoff.get(key) is not False:
                raise SystemExit(f"after_action_learning_packet direct handoff unsafe {key}: {direct_learning.metadata}")
        if (
            direct_learning.metadata.get("after_action_learning_authorizes_execution") is not False
            or direct_learning.metadata.get("after_action_learning_authorizes_completion_claim") is not False
            or direct_learning.metadata.get("after_action_learning_approval_granted") is not False
        ):
            raise SystemExit(f"after_action_learning_packet direct flat authority fields should be false: {direct_learning.metadata}")
        if direct_learning.metadata.get("operator_timeboxes_override_priority") is not True or direct_learning.metadata.get("stop_times_override_priority") is not True:
            raise SystemExit(f"after_action_learning_packet direct call missed operator-limit metadata: {direct_learning.metadata}")
        for key in [
            "calls_model",
            "executes_tools",
            "queues_approval",
            "approves_request",
            "dismisses_request",
            "reads_private_data",
            "writes_files",
            "writes_notes",
            "writes_memory",
            "controls_computer",
            "external_side_effect",
            "requires_approval",
        ]:
            if direct_learning.metadata.get(key) is not False:
                raise SystemExit(f"after_action_learning_packet unsafe metadata {key}: {direct_learning.metadata}")

        planned_learning = runtime.handle("after action learning packet 2")
        if not planned_learning.verified or planned_learning.tool_results[0].tool_name != "after_action_learning_packet":
            raise SystemExit(f"Planner should route after-action learning packets: {planned_learning.tool_results}")
        if planned_learning.tool_results[0].metadata.get("run_id") != 2:
            raise SystemExit(f"Planner routed after-action learning packet with wrong run id: {planned_learning.tool_results[0].metadata}")

        exact_receipt = verification_receipt({"run_id": 2, "expectation": "approval should be held"})
        if not exact_receipt.ok or exact_receipt.metadata.get("run_id") != 2:
            raise SystemExit(f"verification_receipt should find exact run ids outside recent-window logic: {exact_receipt.metadata}")
        if "Tool run #2 was not found" in exact_receipt.output:
            raise SystemExit("verification_receipt should use exact audit lookup for explicit run ids.")

        exact_recovery_small_window = execution_recovery_packet({"run_id": 2, "limit": 1})
        if not exact_recovery_small_window.ok or exact_recovery_small_window.metadata.get("run_id") != 2:
            raise SystemExit(f"execution_recovery_packet should find explicit run ids even with a tiny inspection window: {exact_recovery_small_window.metadata}")
        if exact_recovery_small_window.metadata.get("inspected_runs") != 1:
            raise SystemExit(f"execution_recovery_packet should still report the bounded context window: {exact_recovery_small_window.metadata}")
        if "Problem run:" not in exact_recovery_small_window.output or "verification receipt 2" not in exact_recovery_small_window.output:
            raise SystemExit("execution_recovery_packet exact lookup output missed recovery guidance.")

        mixed_output_run_id = runtime.store.log_tool_run(
            runtime.session_id,
            "run_shell_command",
            "HIGH_RISK",
            False,
            False,
            "Blocked after runtime trace receipt 99998; Safety receipt: queued as approval #77.",
        )
        mixed_recovery = execution_recovery_packet({"run_id": mixed_output_run_id})
        if not mixed_recovery.ok:
            raise SystemExit(f"execution_recovery_packet should inspect mixed-output run: {mixed_recovery.metadata}")
        if mixed_recovery.metadata.get("output_approval_ids") != [77]:
            raise SystemExit(
                f"execution_recovery_packet should infer approval #77 from the approval marker, not runtime trace #99998: {mixed_recovery.metadata}"
            )
        if 99998 in mixed_recovery.metadata.get("recovery_approval_ids", []):
            raise SystemExit(f"execution_recovery_packet should not treat runtime trace #99998 as an approval id: {mixed_recovery.metadata}")
        if "approval packet 77" not in mixed_recovery.output:
            raise SystemExit("execution_recovery_packet mixed output missed the real approval packet id.")
        unknown_tool_run_id = runtime.store.log_tool_run(
            runtime.session_id,
            "missing_tool",
            "UNKNOWN",
            False,
            False,
            '"Unknown tool: missing_tool"',
            metadata={
                "failure_kind": "unknown_tool",
                "executed_handler": False,
                "requires_confirmation": False,
                "planned_arg_keys": ["x"],
            },
        )
        unknown_recovery = execution_recovery_packet({"run_id": unknown_tool_run_id})
        if not unknown_recovery.ok or unknown_recovery.metadata.get("verdict") != "UNKNOWN_TOOL":
            raise SystemExit(f"execution_recovery_packet should classify unknown-tool audit metadata: {unknown_recovery.metadata}")
        for expected in [
            "failure kind: unknown_tool",
            "handler executed: no",
            "planned arg keys: x",
            "route/registry mismatch",
        ]:
            if expected not in unknown_recovery.output:
                raise SystemExit(f"unknown-tool recovery output missed expected text: {expected}")
        tool_error_run_id = runtime.store.log_tool_run(
            runtime.session_id,
            "explode",
            "READ_ONLY",
            False,
            False,
            "Tool error: boom",
            metadata={
                "failure_kind": "tool_error",
                "executed_handler": True,
                "requires_confirmation": False,
                "planned_arg_keys": ["mode"],
                "error_type": "RuntimeError",
            },
        )
        tool_error_recovery = execution_recovery_packet({"run_id": tool_error_run_id})
        if not tool_error_recovery.ok or tool_error_recovery.metadata.get("verdict") != "TOOL_ERROR":
            raise SystemExit(f"execution_recovery_packet should classify handler-error audit metadata: {tool_error_recovery.metadata}")
        for expected in [
            "failure kind: tool_error",
            "handler executed: yes",
            "planned arg keys: mode",
            "handler error",
        ]:
            if expected not in tool_error_recovery.output:
                raise SystemExit(f"tool-error recovery output missed expected text: {expected}")

        empty_runtime = make_temp_runtime(Path(temp) / "empty")
        empty_trace = make_audit_tools(empty_runtime.store)[2]({})
        if empty_trace.ok or empty_trace.metadata.get("verdict") != "MISSING_TRACE":
            raise SystemExit("runtime_trace_receipt should fail cleanly when no trace exists.")
        empty_recovery = make_audit_tools(empty_runtime.store)[4]({})
        if not empty_recovery.ok or empty_recovery.metadata.get("verdict") != "NO_RECENT_RUNS":
            raise SystemExit("execution_recovery_packet should handle empty audit logs safely.")

        bad_recovery = execution_recovery_packet({"run_id": "run-abc"})
        if bad_recovery.ok or bad_recovery.metadata.get("raw_run_id") != "run-abc":
            raise SystemExit(f"execution_recovery_packet should preserve bounded raw run id: {bad_recovery.metadata}")
        path_bad_recovery = execution_recovery_packet({"run_id": "/private/tmp/jarvis-recovery-run-id"})
        if path_bad_recovery.ok or path_bad_recovery.metadata.get("raw_run_id") != "<local-path>":
            raise SystemExit(f"execution_recovery_packet should redact path-shaped bad run ids: {path_bad_recovery.metadata}")

        healthy_runtime = make_temp_runtime(Path(temp) / "healthy")
        healthy_runtime.handle("calculate 2 + 2")
        healthy_runtime.handle("verification receipt latest")
        healthy_health = healthy_runtime.handle("execution health report")
        healthy_metadata = healthy_health.tool_results[0].metadata
        healthy_proof_queue = healthy_metadata.get("execution_proof_queue", [])
        if healthy_metadata.get("verdict") != "HEALTHY_WITH_AUDIT_EVIDENCE":
            raise SystemExit(f"healthy execution_health_report should stay healthy with evidence: {healthy_metadata}")
        if healthy_metadata.get("next_command") != healthy_metadata.get("next_proof_command"):
            raise SystemExit(f"healthy execution_health_report should route next command to first proof command: {healthy_metadata}")
        if healthy_metadata.get("next_command") != healthy_metadata.get("next_required_command"):
            raise SystemExit(f"healthy execution_health_report should expose command-first next required metadata: {healthy_metadata}")
        if healthy_proof_queue and healthy_metadata.get("next_command") != healthy_proof_queue[0]:
            raise SystemExit(f"healthy execution_health_report next command should equal first proof queue item: {healthy_metadata}")
        if healthy_metadata.get("learning_followup_command") != "after-action learning packet 1":
            raise SystemExit(f"healthy execution_health_report should preserve learning as follow-up: {healthy_metadata}")
        healthy_handoff = healthy_metadata.get("execution_health_handoff") or {}
        if healthy_handoff.get("next_command") != healthy_metadata.get("next_command") or healthy_handoff.get("learning_followup_command") != healthy_metadata.get("learning_followup_command"):
            raise SystemExit(f"healthy execution_health_report handoff should mirror command-first fields: {healthy_metadata}")
        if "command-first rule:" not in healthy_health.response:
            raise SystemExit("healthy execution_health_report should explain the command-first rule.")

        bad_after_action = after_action_learning_packet({"run_id": "run-abc"})
        if bad_after_action.ok or bad_after_action.metadata.get("raw_run_id") != "run-abc":
            raise SystemExit(f"after_action_learning_packet should preserve bounded raw run id: {bad_after_action.metadata}")
        path_bad_after_action = after_action_learning_packet({"run_id": "/\x55sers/example/private/after-action-run-id"})
        if path_bad_after_action.ok or path_bad_after_action.metadata.get("raw_run_id") != "<local-path>":
            raise SystemExit(f"after_action_learning_packet should redact path-shaped bad run ids: {path_bad_after_action.metadata}")

        bad_learning_closure = execution_learning_closure_packet({"run_id": "run-abc"})
        if bad_learning_closure.ok or bad_learning_closure.metadata.get("raw_run_id") != "run-abc":
            raise SystemExit(f"execution_learning_closure_packet should preserve bounded raw run id: {bad_learning_closure.metadata}")
        path_bad_learning_closure = execution_learning_closure_packet({"run_id": "/private/tmp/jarvis-learning-run-id"})
        if path_bad_learning_closure.ok or path_bad_learning_closure.metadata.get("raw_run_id") != "<local-path>":
            raise SystemExit(f"execution_learning_closure_packet should redact path-shaped bad run ids: {path_bad_learning_closure.metadata}")

        for tool_name, result in [
            ("execution_recovery_packet", execution_recovery_packet({"run_id": 0})),
            ("after_action_learning_packet", after_action_learning_packet({"run_id": -1})),
            ("execution_learning_closure_packet", execution_learning_closure_packet({"run_id": 0})),
        ]:
            if result.ok or result.metadata.get("verdict") != "INVALID_ID":
                raise SystemExit(f"{tool_name} should reject non-positive run ids before lookup: {result.metadata}")
            if "positive number" not in result.output:
                raise SystemExit(f"{tool_name} should explain positive run id requirement: {result.output}")
            if (
                result.metadata.get("executes_tools")
                or result.metadata.get("queues_approval")
                or result.metadata.get("approves_request")
                or result.metadata.get("dismisses_request")
                or result.metadata.get("reads_private_data")
                or result.metadata.get("writes_files")
                or result.metadata.get("writes_notes")
                or result.metadata.get("writes_memory")
                or result.metadata.get("controls_computer")
            ):
                raise SystemExit(f"{tool_name} invalid-id path should remain read-only: {result.metadata}")


if __name__ == "__main__":
    main()
