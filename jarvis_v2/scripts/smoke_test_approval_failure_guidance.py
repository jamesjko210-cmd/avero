from __future__ import annotations

import re
from pathlib import Path
from tempfile import TemporaryDirectory

import jarvis_v2.tools.approvals as approvals_module
from jarvis_v2.scripts.test_runtime import make_temp_runtime


APPROVAL_HANDLERS = (
    "dismiss_pending_approval",
    "inspect_pending_approval",
    "approval_execution_packet",
    "approval_resume_packet",
    "approval_chain_proof",
    "approval_readiness_packet",
    "approve_pending_approval",
)
PRIVATE_MARKER = "/\x55sers/the operator/private-approval-marker"


def _assert_canonical_failure(result, *, label: str, expected_action: str) -> None:
    if result.ok:
        raise SystemExit(f"{label} unexpectedly succeeded: {result}")
    guidance = result.metadata.get("recovery_guidance")
    if not isinstance(guidance, dict) or guidance.get("version") != 1:
        raise SystemExit(f"{label} missed canonical recovery guidance: {result.metadata}")
    if guidance.get("action") != expected_action or expected_action not in result.output:
        raise SystemExit(f"{label} exposed the wrong recovery action: {result}")
    for command in guidance.get("commands") or []:
        if command not in result.output:
            raise SystemExit(f"{label} hid declared command {command!r}: {result.output}")
    if result.metadata.get("authorizes_execution") is not False:
        raise SystemExit(f"{label} authorized execution: {result.metadata}")
    if result.metadata.get("approval_granted") is not False:
        raise SystemExit(f"{label} granted approval: {result.metadata}")
    if (
        result.metadata.get("outcome_known") is not True
        or result.metadata.get("side_effect_possible") is not False
        or result.metadata.get("automatic_retry_allowed") is not False
        or result.metadata.get("authorizes_retry") is not False
    ):
        raise SystemExit(f"{label} missed known side-effect-free refusal boundaries: {result.metadata}")
    if result.metadata.get("writes_files") or result.metadata.get("writes_memory"):
        raise SystemExit(f"{label} claimed a write: {result.metadata}")
    combined = f"{result.output}\n{result.metadata}"
    if PRIVATE_MARKER in combined or re.search(r"/(?:Users|private|var/folders|tmp)/", combined):
        raise SystemExit(f"{label} leaked private local detail: {combined}")


def test_invalid_and_missing_ids() -> None:
    with TemporaryDirectory(prefix="jarvis-approval-guidance-") as temp:
        runtime = make_temp_runtime(Path(temp))
        for tool_name in APPROVAL_HANDLERS:
            handler = runtime.registry.get(tool_name).handler
            invalid = handler({"approval_id": PRIVATE_MARKER})
            _assert_canonical_failure(
                invalid,
                label=f"{tool_name} invalid id",
                expected_action=approvals_module.APPROVAL_ID_RECOVERY_ACTION,
            )
            if invalid.metadata.get("reason") != "bad_approval_id":
                raise SystemExit(f"{tool_name} invalid id lost its stable reason: {invalid.metadata}")

            missing = handler({"approval_id": 987654321})
            _assert_canonical_failure(
                missing,
                label=f"{tool_name} missing id",
                expected_action=approvals_module.RESOURCE_NOT_FOUND_RECOVERY_ACTION,
            )
            if missing.metadata.get("reason") != "not_found":
                raise SystemExit(f"{tool_name} missing id lost its stable reason: {missing.metadata}")
            if missing.metadata.get("next_command") != "pending approvals":
                raise SystemExit(f"{tool_name} missing id did not refresh the queue first: {missing.metadata}")


def _new_pending(runtime, suffix: str) -> int:
    return runtime.store.add_pending_approval(
        runtime.session_id,
        f"approval guidance {suffix}",
        "run_shell_command",
        "approval required",
        planned_args={"command": f"printf {suffix}"},
    )


def test_stale_and_changed_review_refusals() -> None:
    with TemporaryDirectory(prefix="jarvis-approval-review-guidance-") as temp:
        runtime = make_temp_runtime(Path(temp))
        execution_packet = runtime.registry.get("approval_execution_packet").handler
        readiness = runtime.registry.get("approval_readiness_packet").handler
        approve = runtime.registry.get("approve_pending_approval").handler

        stale_id = _new_pending(runtime, "stale")
        original_staleness = approvals_module._approval_staleness_label
        approvals_module._approval_staleness_label = lambda _age: "stale"
        try:
            stale = execution_packet({"approval_id": stale_id})
        finally:
            approvals_module._approval_staleness_label = original_staleness
        _assert_canonical_failure(
            stale,
            label="stale last-look packet",
            expected_action=approvals_module.APPROVAL_REVIEW_RECOVERY_ACTION,
        )
        if stale.metadata.get("reason") != "readiness_required":
            raise SystemExit(f"stale last-look refusal lost readiness state: {stale.metadata}")

        no_readiness_id = _new_pending(runtime, "no-readiness")
        no_readiness = approve({"approval_id": no_readiness_id})
        _assert_canonical_failure(
            no_readiness,
            label="approve without readiness",
            expected_action=approvals_module.APPROVAL_REVIEW_RECOVERY_ACTION,
        )

        packet_required_id = _new_pending(runtime, "packet-required")
        ready = readiness({"approval_id": packet_required_id})
        if not ready.ok or ready.metadata.get("approval_readiness_receipt_issued") is not True:
            raise SystemExit(f"could not prepare last-look refusal: {ready}")
        packet_required = approve({"approval_id": packet_required_id})
        _assert_canonical_failure(
            packet_required,
            label="approve without last-look packet",
            expected_action=approvals_module.APPROVAL_REVIEW_RECOVERY_ACTION,
        )

        changed_id = _new_pending(runtime, "changed")
        ready = readiness({"approval_id": changed_id})
        packet = execution_packet({"approval_id": changed_id})
        if not ready.ok or not packet.ok:
            raise SystemExit(f"could not prepare changed-readiness refusal: {ready} / {packet}")
        store_type = type(runtime.store)
        original_approve = store_type.approve_pending_approval_if_ready
        store_type.approve_pending_approval_if_ready = lambda *_args, **_kwargs: False
        try:
            changed = approve({"approval_id": changed_id})
        finally:
            store_type.approve_pending_approval_if_ready = original_approve
        _assert_canonical_failure(
            changed,
            label="approval changed after readiness",
            expected_action=approvals_module.APPROVAL_REVIEW_RECOVERY_ACTION,
        )
        row = runtime.store.get_approval(changed_id)
        if row is None or str(row["status"]).lower() != "pending":
            raise SystemExit("changed-readiness refusal mutated the pending approval")


def main() -> None:
    test_invalid_and_missing_ids()
    test_stale_and_changed_review_refusals()
    print("Approval failure-guidance smoke test passed (19 refusal routes pinned).")


if __name__ == "__main__":
    main()
