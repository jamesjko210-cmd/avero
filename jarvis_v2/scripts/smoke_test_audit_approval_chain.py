from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from jarvis_v2.memory.store import approval_action_digest
from jarvis_v2.scripts.test_runtime import make_temp_runtime
from jarvis_v2.tools.audit import make_audit_tools


ACTION_TOOL = "audit_chain_action"
ACTION_ARGS = {"target": "synthetic"}
CLAIMED_AT = "2026-01-01T00:00:00+00:00"
COMPLETED_AT = "2026-01-01T00:00:01+00:00"


def _add_approval(
    runtime,
    label: str,
    *,
    status: str = "approved",
    tool_name: str = ACTION_TOOL,
    planned_args: dict | None = None,
    claim_outcome: str | None = "succeeded",
    completed: bool = True,
) -> int:
    args = ACTION_ARGS if planned_args is None else planned_args
    approval_id = runtime.store.add_pending_approval(
        runtime.session_id,
        f"{label} synthetic approval",
        tool_name,
        "synthetic approval-chain audit fixture",
        planned_args=args,
    )
    with runtime.store.connect() as conn:
        conn.execute(
            "UPDATE pending_approvals SET status = ? WHERE id = ?",
            (status, approval_id),
        )
        if claim_outcome is not None:
            conn.execute(
                """
                INSERT INTO approval_execution_claims(
                    approval_id, claim_token, outcome, claimed_at, completed_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    approval_id,
                    f"synthetic-{label}",
                    claim_outcome,
                    CLAIMED_AT,
                    COMPLETED_AT if completed else None,
                ),
            )
    return approval_id


def _log_risky_run(
    runtime,
    approval_id: int | None,
    *,
    digest: str | None = None,
    tool_name: str = ACTION_TOOL,
    risk: str = "HIGH_RISK",
    ok: bool = True,
    metadata: dict | None = None,
) -> int:
    return runtime.store.log_tool_run(
        runtime.session_id,
        tool_name,
        risk,
        ok,
        True,
        f"synthetic risky action reported {'success' if ok else 'failure'}",
        approval_id=approval_id,
        approval_action_digest_value=(
            approval_action_digest(tool_name, ACTION_ARGS) if digest is None else digest
        ),
        metadata=metadata,
    )


def _make_invalid_case(runtime, case: str) -> int:
    if case == "missing":
        return _log_risky_run(runtime, None)
    if case == "nonexistent":
        return _log_risky_run(runtime, 999_999)
    if case == "unrelated":
        approval_id = _add_approval(runtime, case, tool_name="unrelated_action")
        return _log_risky_run(runtime, approval_id)
    if case == "wrong_digest":
        approval_id = _add_approval(runtime, case)
        return _log_risky_run(runtime, approval_id, digest="0" * 64)
    if case in {"pending", "dismissed", "invalid_status"}:
        status = {"pending": "pending", "dismissed": "dismissed", "invalid_status": "unexpected"}[case]
        approval_id = _add_approval(runtime, case, status=status)
        return _log_risky_run(runtime, approval_id)
    if case == "unfinished":
        approval_id = _add_approval(
            runtime,
            case,
            claim_outcome="claimed",
            completed=False,
        )
        return _log_risky_run(runtime, approval_id)
    if case == "failed":
        approval_id = _add_approval(runtime, case, claim_outcome="failed")
        return _log_risky_run(runtime, approval_id)
    if case == "duplicate":
        approval_id = _add_approval(runtime, case)
        first_run_id = _log_risky_run(runtime, approval_id)
        _log_risky_run(runtime, approval_id)
        return first_run_id
    if case == "malformed":
        approval_id = _add_approval(runtime, case)
        with runtime.store.connect() as conn:
            conn.execute(
                "UPDATE pending_approvals SET planned_args = ? WHERE id = ?",
                ("{malformed", approval_id),
            )
        return _log_risky_run(runtime, approval_id)
    if case == "malformed_risk":
        return _log_risky_run(runtime, None, risk="surprise-risk")
    if case == "forged_failure":
        return _log_risky_run(runtime, 999_999, ok=False)
    if case == "forged_unknown":
        return _log_risky_run(
            runtime,
            999_999,
            ok=False,
            metadata={"execution_outcome_unknown": True},
        )
    raise AssertionError(f"unknown synthetic case: {case}")


def _assert_invalid_consumers(runtime, run_id: int, case: str) -> None:
    (
        recent_tool_runs,
        verification_receipt,
        _,
        execution_audit_gate,
        execution_recovery_packet,
        after_action_learning_packet,
        execution_health_report,
        recovery_closure_checklist,
        execution_learning_closure_packet,
    ) = make_audit_tools(runtime.store)

    recent = recent_tool_runs({"limit": 200})
    recent_rows = recent.metadata.get("recent_tool_run_rows") or []
    target_summary = next((row for row in recent_rows if row.get("run_id") == run_id), None)
    if (
        target_summary is None
        or target_summary.get("status") != "failed"
        or target_summary.get("approved") is not False
    ):
        raise SystemExit(f"{case}: recent_tool_runs trusted an invalid chain: {recent.metadata}")

    receipt = verification_receipt({"run_id": run_id})
    if (
        receipt.metadata.get("verdict") != "APPROVAL_EVIDENCE_MISSING"
        or receipt.metadata.get("approved") is not False
    ):
        raise SystemExit(f"{case}: verification_receipt trusted an invalid chain: {receipt.metadata}")

    gate = execution_audit_gate({"limit": 200})
    if gate.metadata.get("review_required") is not True or gate.metadata.get(
        "safe_to_trust_recent_execution"
    ) is not False:
        raise SystemExit(f"{case}: execution_audit_gate cleared an invalid chain: {gate.metadata}")

    recovery = execution_recovery_packet({"run_id": run_id, "limit": 200})
    if (
        recovery.metadata.get("verdict")
        != (
            "APPROVAL_LINK_MISSING"
            if case in {"missing", "malformed_risk"}
            else "APPROVAL_EVIDENCE_MISSING"
        )
        or recovery.metadata.get("problem_found") is not True
        or recovery.metadata.get("approved") is not False
    ):
        raise SystemExit(f"{case}: execution_recovery_packet cleared an invalid chain: {recovery.metadata}")

    learning = after_action_learning_packet({"run_id": run_id, "limit": 200})
    if (
        learning.metadata.get("verdict") != "PROVE_APPROVAL_CHAIN_BEFORE_LEARNING"
        or learning.metadata.get("approval_problem") is not True
        or learning.metadata.get("approved") is not False
    ):
        raise SystemExit(f"{case}: after_action_learning_packet trusted an invalid chain: {learning.metadata}")

    health = execution_health_report({"limit": 200})
    if (
        health.metadata.get("verdict") != "APPROVAL_REVIEW_REQUIRED"
        or health.metadata.get("risky_approval_problems", 0) < 1
        or health.metadata.get("review_required") is not True
        or health.metadata.get("safe_to_continue") is not False
    ):
        raise SystemExit(f"{case}: execution_health_report cleared an invalid chain: {health.metadata}")

    recovery_closure = recovery_closure_checklist({"limit": 200})
    if (
        recovery_closure.metadata.get("blocks_completion_claim") is not True
        or recovery_closure.metadata.get("ready_to_retry") is not False
        or "approval_chain_proof" not in (recovery_closure.metadata.get("missing") or [])
    ):
        raise SystemExit(f"{case}: recovery closure cleared an invalid chain: {recovery_closure.metadata}")

    learning_closure = execution_learning_closure_packet({"run_id": run_id, "limit": 200})
    if (
        learning_closure.metadata.get("learning_closure_ready") is not False
        or learning_closure.metadata.get("learning_closure_blocks_completion_claim") is not True
    ):
        raise SystemExit(f"{case}: learning closure cleared an invalid chain: {learning_closure.metadata}")


def _log_meta_packets(runtime, run_id: int, *, risk: str = "READ_ONLY") -> None:
    packets = (
        ("verification_receipt", "verification_receipt_handoff"),
        ("execution_recovery_packet", "execution_recovery_handoff"),
        ("after_action_learning_packet", "after_action_learning_handoff"),
    )
    for tool_name, handoff_key in packets:
        runtime.store.log_tool_run(
            runtime.session_id,
            tool_name,
            risk,
            True,
            True,
            f"synthetic {tool_name} meta receipt",
            approval_id=999_998,
            metadata={
                "run_id": run_id,
                handoff_key: {
                    "source": tool_name,
                    "target": {"run_id": run_id},
                },
            },
        )


def _assert_forged_meta_receipts_do_not_close(root: Path) -> None:
    runtime = make_temp_runtime(root)
    run_id = _make_invalid_case(runtime, "nonexistent")
    _log_meta_packets(runtime, run_id)
    tools = make_audit_tools(runtime.store)
    health = tools[6]({"limit": 200})
    if (
        health.metadata.get("recovery_closure_target_verification_receipts") != 0
        or health.metadata.get("recovery_closure_target_recovery_packets") != 0
        or health.metadata.get("recovery_closure_target_after_action_learning_packets") != 0
    ):
        raise SystemExit(f"forged meta receipts received closure credit: {health.metadata}")
    if tools[7]({"limit": 200}).metadata.get("blocks_completion_claim") is not True:
        raise SystemExit("forged meta receipts cleared recovery closure")
    learning_closure = tools[8]({"run_id": run_id, "limit": 200})
    if learning_closure.metadata.get("learning_closure_ready") is not False:
        raise SystemExit(f"forged meta receipts cleared learning closure: {learning_closure.metadata}")


def _assert_baseline(root: Path, *, risky: bool, risk: str = "HIGH_RISK") -> None:
    runtime = make_temp_runtime(root)
    if risky:
        approval_id = _add_approval(runtime, "valid")
        run_id = _log_risky_run(runtime, approval_id, risk=risk)
        label = "valid risky"
    else:
        run_id = runtime.store.log_tool_run(
            runtime.session_id,
            "low_risk_action",
            "LOCAL_SAFE",
            True,
            True,
            "synthetic low-risk success",
            approval_id=999_997,
        )
        label = "low risk"
    _log_meta_packets(runtime, run_id)
    tools = make_audit_tools(runtime.store)

    recent_rows = tools[0]({"limit": 200}).metadata.get("recent_tool_run_rows") or []
    target_summary = next(row for row in recent_rows if row.get("run_id") == run_id)
    if target_summary.get("status") != "ok" or target_summary.get("approved") is not True:
        raise SystemExit(f"{label} baseline changed in recent_tool_runs: {target_summary}")
    receipt = tools[1]({"run_id": run_id})
    if receipt.metadata.get("verdict") != "PASS_WITH_AUDIT_EVIDENCE" or receipt.metadata.get("approved") is not True:
        raise SystemExit(f"{label} baseline changed in verification_receipt: {receipt.metadata}")
    gate = tools[3]({"limit": 200})
    if gate.metadata.get("review_required") is not False or gate.metadata.get("safe_to_trust_recent_execution") is not True:
        raise SystemExit(f"{label} baseline changed in execution_audit_gate: {gate.metadata}")
    recovery = tools[4]({"run_id": run_id, "limit": 200})
    if recovery.metadata.get("verdict") != "RECOVERY_NOT_REQUIRED":
        raise SystemExit(f"{label} baseline changed in execution_recovery_packet: {recovery.metadata}")
    learning = tools[5]({"run_id": run_id, "limit": 200})
    if learning.metadata.get("verdict") != "SAFE_TO_REVIEW_FOR_LEARNING":
        raise SystemExit(f"{label} baseline changed in after_action_learning_packet: {learning.metadata}")
    health = tools[6]({"limit": 200})
    if health.metadata.get("risky_approval_problems") != 0 or health.metadata.get("review_required") is not False:
        raise SystemExit(f"{label} baseline changed in execution_health_report: {health.metadata}")
    if tools[7]({"limit": 200}).metadata.get("blocks_completion_claim") is not False:
        raise SystemExit(f"{label} baseline changed in recovery closure")
    closure = tools[8]({"run_id": run_id, "limit": 200})
    if closure.metadata.get("learning_closure_ready") is not True:
        raise SystemExit(f"{label} baseline changed in learning closure: {closure.metadata}")


def _assert_exact_non_success(root: Path, *, outcome_unknown: bool) -> None:
    runtime = make_temp_runtime(root)
    label = "exact unknown" if outcome_unknown else "exact failure"
    approval_id = _add_approval(
        runtime,
        label,
        claim_outcome="outcome_unknown" if outcome_unknown else "failed",
    )
    run_id = _log_risky_run(
        runtime,
        approval_id,
        ok=False,
        metadata={"execution_outcome_unknown": True} if outcome_unknown else {},
    )
    tools = make_audit_tools(runtime.store)
    recent_rows = tools[0]({"limit": 200}).metadata.get("recent_tool_run_rows") or []
    target = next(row for row in recent_rows if row.get("run_id") == run_id)
    if target.get("status") != "failed" or target.get("approved") is not True:
        raise SystemExit(f"{label}: exact linkage was not reported truthfully: {target}")
    receipt = tools[1]({"run_id": run_id})
    receipt_handoff = receipt.metadata.get("verification_receipt_handoff") or {}
    if (
        receipt.metadata.get("verdict") != "FAILED_OR_BLOCKED"
        or receipt.metadata.get("approved") is not True
        or receipt_handoff.get("approval_problem") is not False
    ):
        raise SystemExit(f"{label}: verification classification changed: {receipt.metadata}")
    recovery = tools[4]({"run_id": run_id, "limit": 200})
    if recovery.metadata.get("verdict") != "FAILED_OR_BLOCKED" or recovery.metadata.get("approved") is not True:
        raise SystemExit(f"{label}: recovery classification changed: {recovery.metadata}")
    learning = tools[5]({"run_id": run_id, "limit": 200})
    if (
        learning.metadata.get("verdict") != "PROMOTE_FAILURE_TO_REVIEW"
        or learning.metadata.get("approval_problem") is not False
        or learning.metadata.get("approved") is not True
    ):
        raise SystemExit(f"{label}: learning classification changed: {learning.metadata}")
    health = tools[6]({"limit": 200})
    if health.metadata.get("risky_approval_problems") != 0 or health.metadata.get("review_required") is not True:
        raise SystemExit(f"{label}: health should separate failure from approval linkage: {health.metadata}")


def _assert_classified_history(root: Path) -> None:
    cases = {
        "valid": ("ok", False),
        "wrong_tool": ("invalid linkage", False),
        "wrong_digest": ("invalid linkage", False),
        "duplicate": ("invalid linkage", False),
        "unfinished": ("outcome unknown, exact approval linkage", True),
    }
    for case, (expected_text, exact_unknown) in cases.items():
        runtime = make_temp_runtime(root / case)
        if case == "valid":
            approval_id = _add_approval(runtime, case)
            _log_risky_run(runtime, approval_id)
        elif case == "wrong_tool":
            approval_id = _add_approval(runtime, case, tool_name="unrelated_action")
            _log_risky_run(runtime, approval_id)
        elif case == "wrong_digest":
            approval_id = _add_approval(runtime, case)
            _log_risky_run(runtime, approval_id, digest="0" * 64)
        elif case == "duplicate":
            approval_id = _add_approval(runtime, case)
            _log_risky_run(runtime, approval_id)
            _log_risky_run(runtime, approval_id)
        else:
            approval_id = _add_approval(runtime, case, claim_outcome="claimed", completed=False)
            _log_risky_run(runtime, approval_id, ok=False, metadata={"execution_outcome_unknown": True})
        history = runtime.registry.get("approval_history").handler({"limit": 10})
        if expected_text not in history.output:
            raise SystemExit(f"{case}: approval history missed classified state: {history.output}")
        expected_count = 1 if case == "valid" or exact_unknown else 0
        if history.metadata.get("approved_tool_runs") != expected_count:
            raise SystemExit(f"{case}: approval history counted raw linkage: {history.metadata}")


def main() -> None:
    cases = (
        "missing",
        "nonexistent",
        "unrelated",
        "wrong_digest",
        "pending",
        "dismissed",
        "duplicate",
        "invalid_status",
        "malformed",
        "malformed_risk",
        "forged_failure",
        "forged_unknown",
    )
    with TemporaryDirectory(prefix="jarvis-audit-approval-chain-") as temp:
        root = Path(temp)
        for case in cases:
            runtime = make_temp_runtime(root / case)
            _assert_invalid_consumers(runtime, _make_invalid_case(runtime, case), case)
        _assert_forged_meta_receipts_do_not_close(root / "forged-meta")
        _assert_baseline(root / "valid", risky=True)
        _assert_baseline(root / "lowercase-risk", risky=True, risk="high_risk")
        _assert_baseline(root / "low-risk", risky=False)
        _assert_exact_non_success(root / "exact-failure", outcome_unknown=False)
        _assert_exact_non_success(root / "exact-unknown", outcome_unknown=True)
        _assert_classified_history(root / "history")
    print("[ok] audit approval-chain adversarial matrix")
    print("[ok] forged meta receipts cannot clear recovery or learning closure")
    print("[ok] exact valid risky, lowercase risky, failure/unknown, and low-risk baselines")
    print("[ok] approval history consumes classified row-bound evidence")


if __name__ == "__main__":
    main()
