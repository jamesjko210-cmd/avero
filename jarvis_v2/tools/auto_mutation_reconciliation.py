from __future__ import annotations

from datetime import datetime, timezone
from threading import Lock
import re
import time
from typing import Any

from jarvis_v2.agent.failure_guidance import declare_failure_guidance
from jarvis_v2.agent.types import ToolResult
from jarvis_v2.memory.store import MemoryStore


MAX_RECONCILIATION_ROWS = 100
ALLOWED_DISPOSITIONS = {"confirmed_applied", "confirmed_not_applied"}
REVIEW_RECEIPT_TTL_SECONDS = 300


def _metadata(**extra: Any) -> dict[str, Any]:
    return {
        "calls_model": False,
        "calls_external_service": False,
        "executes_tools": False,
        "reads_private_data": False,
        "reads_personal_data": False,
        "executes_side_effect": False,
        "external_side_effect": False,
        "writes_files": False,
        "writes_database": False,
        "writes_memory": False,
        "writes_notes": False,
        "queues_approval": False,
        "requires_approval": False,
        "controls_computer": False,
        "authorizes_retry": False,
        "authorizes_execution": False,
        "authorizes_completion_claim": False,
        "approval_granted": False,
        "content_in_metadata": False,
        "content_exposed": False,
        "target_locator_available": False,
        "execution_outcome_proven": False,
        "approval_proof_affected": False,
        "mutation_rerun_performed": False,
        "changed_state": False,
        "request_digest_exposed": False,
        "action_digest_exposed": False,
        "uncertainty_digest_exposed": False,
        "run_token_exposed": False,
        **extra,
    }


def _reconciliation_failure_metadata(
    *,
    output: str,
    action: str,
    commands: tuple[str, ...],
    failure_kind: str,
    **extra: Any,
) -> dict[str, Any]:
    extra.setdefault("mutation_rerun", False)
    return declare_failure_guidance(
        _metadata(
            failure_kind=failure_kind,
            retry_safe=False,
            automatic_retry_allowed=False,
            retry_requires_reconciliation_review=True,
            **extra,
        ),
        output=output,
        action=action,
        commands=commands,
    )


def _limit(value: Any, default: int = 25) -> int:
    if type(value) is not int:
        return default
    return max(1, min(value, MAX_RECONCILIATION_ROWS))


def _row_payload(row: Any) -> dict[str, Any]:
    disposition = str(row["reconciliation_disposition"] or "")
    return {
        "receipt_id": int(row["id"]),
        "tool_name": str(row["tool_name"]),
        "action_index": int(row["action_index"]),
        "uncertainty_origin": str(row["resolution"]),
        "uncertain_at": str(row["uncertain_at"] or ""),
        "reconciliation_disposition": disposition,
        "resolved_at": str(row["resolved_at"] or ""),
        "resolved": bool(disposition),
    }


def make_auto_mutation_reconciliation_tools(store: MemoryStore, session_id: str):
    review_receipts: dict[int, tuple[str, str, float]] = {}
    review_lock = Lock()

    def auto_mutation_reconciliation_queue(args: dict[str, Any]) -> ToolResult:
        limit = _limit(args.get("limit"))
        rows = store.list_auto_mutation_reconciliation_receipts(limit=limit)
        payloads = [_row_payload(row) for row in rows]
        if not payloads:
            output = "No unresolved uncertain local mutations need operator reconciliation."
        else:
            lines = ["Uncertain local mutations awaiting reconciliation:"]
            for item in payloads:
                lines.append(
                    f"- receipt #{item['receipt_id']} | {item['tool_name']} | "
                    f"origin: {item['uncertainty_origin']} | uncertain: {item['uncertain_at']}"
                )
            lines.extend(
                [
                    "",
                    "Review the real target state before resolving a receipt.",
                    "First inspect it with `uncertain mutation <id>`.",
                    "- If the mutation is present: `resolve mutation receipt <id> as applied`",
                    "- If the mutation is absent: `resolve mutation receipt <id> as not applied`",
                    "Resolution records evidence only; it never reruns the original request.",
                ]
            )
            output = "\n".join(lines)
        return ToolResult(
            "auto_mutation_reconciliation_queue",
            True,
            output,
            _metadata(
                count=len(payloads),
                limit=limit,
                receipts=payloads,
                unresolved_only=True,
                mutation_rerun=False,
            ),
        )

    def auto_mutation_reconciliation_status(args: dict[str, Any]) -> ToolResult:
        receipt_id = args.get("receipt_id")
        if type(receipt_id) is not int or receipt_id <= 0:
            return ToolResult(
                "auto_mutation_reconciliation_status",
                False,
                "Receipt id must be a positive integer.",
                _metadata(failure_kind="bad_receipt_id", receipt_id=None),
            )
        row = store.get_auto_mutation_reconciliation_receipt(receipt_id)
        if row is None:
            return ToolResult(
                "auto_mutation_reconciliation_status",
                False,
                f"Mutation receipt #{receipt_id} was not found.",
                _metadata(failure_kind="receipt_not_found", receipt_id=receipt_id),
            )
        item = _row_payload(row)
        if str(row["state"]) != "uncertain" or str(row["result"]) != "unknown":
            output = (
                f"Mutation receipt #{receipt_id} is not uncertain and does not need reconciliation. "
                "No action was taken."
            )
            eligible = False
        elif item["resolved"]:
            output = (
                f"Mutation receipt #{receipt_id} was reconciled as "
                f"{item['reconciliation_disposition'].replace('_', ' ')}. The original request was not rerun."
            )
            eligible = False
        else:
            uncertainty_digest = row["uncertainty_digest"]
            if (
                type(uncertainty_digest) is not str
                or re.fullmatch(r"[0-9a-f]{64}", uncertainty_digest) is None
            ):
                recovery_action = (
                    "This uncertain receipt has malformed proof. Run `auto mutation reconciliation "
                    "queue`, then `recent tool runs` and `setup check`; repair storage before "
                    "resolving. Do not resolve or rerun the mutation."
                )
                return ToolResult(
                    "auto_mutation_reconciliation_status",
                    False,
                    recovery_action,
                    _reconciliation_failure_metadata(
                        output=recovery_action,
                        action=recovery_action,
                        commands=(
                            "auto mutation reconciliation queue",
                            "recent tool runs",
                            "setup check",
                        ),
                        failure_kind="malformed_uncertainty_proof",
                        receipt_id=receipt_id,
                        reconciliation_eligible=False,
                    ),
                )
            reviewed_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
            with review_lock:
                review_receipts[receipt_id] = (
                    uncertainty_digest,
                    reviewed_at,
                    time.monotonic() + REVIEW_RECEIPT_TTL_SECONDS,
                )
            output = (
                f"Mutation receipt #{receipt_id} for {item['tool_name']} is outcome-uncertain "
                f"({item['uncertainty_origin']}). Review the target state, then resolve it as applied or not applied. "
                "Inspection does not rerun the mutation."
            )
            eligible = True
        return ToolResult(
            "auto_mutation_reconciliation_status",
            True,
            output,
            _metadata(
                receipt=item,
                receipt_id=receipt_id,
                reconciliation_eligible=eligible,
                review_receipt_created=eligible,
                review_receipt_expires_seconds=REVIEW_RECEIPT_TTL_SECONDS if eligible else 0,
                review_receipt_private=True,
                mutation_rerun=False,
            ),
        )

    def resolve_auto_mutation_receipt(args: dict[str, Any]) -> ToolResult:
        receipt_id = args.get("receipt_id")
        disposition = args.get("disposition")
        if type(receipt_id) is not int or receipt_id <= 0:
            return ToolResult(
                "resolve_auto_mutation_receipt",
                False,
                "Receipt id must be a positive integer. Nothing changed.",
                _metadata(failure_kind="bad_receipt_id", receipt_id=None),
            )
        if type(disposition) is not str or disposition not in ALLOWED_DISPOSITIONS:
            return ToolResult(
                "resolve_auto_mutation_receipt",
                False,
                "Disposition must be confirmed_applied or confirmed_not_applied. Nothing changed.",
                _metadata(
                    failure_kind="bad_reconciliation_disposition",
                    receipt_id=receipt_id,
                    allowed_dispositions=sorted(ALLOWED_DISPOSITIONS),
                ),
            )
        with review_lock:
            review_receipt = review_receipts.pop(receipt_id, None)
        if review_receipt is None or review_receipt[2] < time.monotonic():
            recovery_action = (
                "Run `uncertain mutation <id>` immediately before resolving. Do not reuse an "
                "expired review or rerun the mutation."
            )
            output = (
                f"Mutation receipt #{receipt_id} has no fresh bound review. {recovery_action} "
                "Nothing changed."
            )
            return ToolResult(
                "resolve_auto_mutation_receipt",
                False,
                output,
                _reconciliation_failure_metadata(
                    output=output,
                    action=recovery_action,
                    commands=("uncertain mutation <id>",),
                    failure_kind="reconciliation_review_required",
                    receipt_id=receipt_id,
                    review_receipt_required=True,
                    mutation_rerun=False,
                ),
            )
        expected_uncertainty_digest, reviewed_at, _expires_at = review_receipt
        success_output = (
            f"Reconciled mutation receipt #{receipt_id} as {disposition.replace('_', ' ')}. "
            "The original request was not rerun."
        )
        atomic_audit_metadata = {
            "executed_handler": True,
            "handler_invoked": True,
            "reconciliation_status": "resolved",
            "reconciliation_disposition": disposition,
            "receipt_id": receipt_id,
            "mutation_rerun": False,
            "authorizes_execution": False,
            "authorizes_completion_claim": False,
            "approval_granted": False,
        }
        status = store.resolve_auto_mutation_receipt(
            receipt_id,
            disposition,
            expected_uncertainty_digest=expected_uncertainty_digest,
            session_id=session_id,
            reviewed_at=reviewed_at,
            output=success_output,
            metadata=atomic_audit_metadata,
        )
        if status.status == "RESOLVED":
            return ToolResult(
                "resolve_auto_mutation_receipt",
                True,
                success_output,
                _metadata(
                    receipt_id=receipt_id,
                    reconciliation_status="resolved",
                    reconciliation_disposition=disposition,
                    writes_database=True,
                    changed_state=True,
                    resolution_recorded=True,
                    same_request_replay_blocked=True,
                    cross_request_fence_active=False,
                    mutation_rerun=False,
                    logged_tool_run_id=status.tool_run_id,
                    audit_finalized_atomically=True,
                ),
            )
        if status.status == "ALREADY_RESOLVED_SAME":
            return ToolResult(
                "resolve_auto_mutation_receipt",
                True,
                (
                    f"Mutation receipt #{receipt_id} was already reconciled as "
                    f"{status.disposition.replace('_', ' ')}. The original request was not rerun."
                ),
                _metadata(
                    failure_kind="",
                    receipt_id=receipt_id,
                    reconciliation_status="already_resolved",
                    reconciliation_disposition=status.disposition,
                    requested_disposition=disposition,
                    idempotent_replay=True,
                    mutation_rerun=False,
                ),
            )
        if status.status == "RESOLUTION_CONFLICT":
            recovery_action = (
                "Run `uncertain mutation <id>` and investigate the recorded disposition. Do not "
                "submit another resolution or rerun the mutation."
            )
            output = (
                f"Mutation receipt #{receipt_id} is already reconciled as "
                f"{status.disposition.replace('_', ' ')}, which conflicts with the requested "
                f"{disposition.replace('_', ' ')}. {recovery_action}"
            )
            return ToolResult(
                "resolve_auto_mutation_receipt",
                False,
                output,
                _reconciliation_failure_metadata(
                    output=output,
                    action=recovery_action,
                    commands=("uncertain mutation <id>",),
                    failure_kind="reconciliation_conflict",
                    receipt_id=receipt_id,
                    reconciliation_status="already_resolved",
                    reconciliation_disposition=status.disposition,
                    requested_disposition=disposition,
                    idempotent_replay=False,
                ),
            )
        failure_kind = {
            "NOT_FOUND": "receipt_not_found",
            "REVIEW_MISMATCH": "reconciliation_review_mismatch",
        }.get(status.status, "receipt_not_uncertain")
        if status.status == "REVIEW_MISMATCH":
            recovery_action = (
                "Run `uncertain mutation <id>` to create a fresh bound review, then decide again. "
                "Do not reuse the stale resolution or rerun the mutation."
            )
            output = (
                f"Mutation receipt #{receipt_id} changed since review. {recovery_action} "
                "Nothing changed."
            )
            metadata = _reconciliation_failure_metadata(
                output=output,
                action=recovery_action,
                commands=("uncertain mutation <id>",),
                failure_kind="reconciliation_review_mismatch",
                receipt_id=receipt_id,
                reconciliation_status=status.status.lower(),
            )
        else:
            output = (
                f"Mutation receipt #{receipt_id} was not found. Nothing changed."
                if status.status == "NOT_FOUND"
                else f"Mutation receipt #{receipt_id} is not an unresolved uncertain receipt. Nothing changed."
            )
            metadata = _metadata(
                failure_kind=failure_kind,
                receipt_id=receipt_id,
                reconciliation_status=status.status.lower(),
                mutation_rerun=False,
            )
        return ToolResult(
            "resolve_auto_mutation_receipt",
            False,
            output,
            metadata,
        )

    return (
        auto_mutation_reconciliation_queue,
        auto_mutation_reconciliation_status,
        resolve_auto_mutation_receipt,
    )
