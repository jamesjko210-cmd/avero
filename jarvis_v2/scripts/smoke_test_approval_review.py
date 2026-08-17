from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from jarvis_v2.scripts.test_runtime import make_temp_runtime


def assert_contains(text: str, expected: list[str], label: str) -> None:
    missing = [item for item in expected if item not in text]
    if missing:
        raise SystemExit(f"{label} missing expected text: {missing}")


def assert_operator_limits(metadata: dict, label: str) -> None:
    if metadata.get("operator_timeboxes_override_priority") is not True or metadata.get("stop_times_override_priority") is not True:
        raise SystemExit(f"{label} missed the operator stop-time/work-window override metadata: {metadata}")


def assert_pending_approvals_path_display(metadata: dict, label: str) -> None:
    path = str(metadata.get("path") or "")
    display = str(metadata.get("path_display") or "")
    if not path.endswith("Jarvis/Automations/Pending Approvals.md"):
        raise SystemExit(f"{label} missed exact pending approvals audit path: {metadata}")
    if display != "Automations/Pending Approvals.md":
        raise SystemExit(f"{label} missed safe pending approvals path display: {metadata}")


def main() -> None:
    with TemporaryDirectory(prefix="jarvis-approval-review-") as temp:
        runtime = make_temp_runtime(Path(temp))
        approval_review_note = Path(temp) / "Vault" / "Jarvis" / "Automations" / "Approval Review.md"
        for case in [
            "run command python3 --version",
            "get clipboard",
            "enable computer control",
            "remind me to review approval safety",
        ]:
            result = runtime.handle(case)
            print(f"[blocked={not result.verified}] {case}")
            print(result.response[:1000])
            print()
            if result.verified:
                raise SystemExit(f"Expected '{case}' to be blocked.")

        for case in ["approval review", "review pending approvals", "should I approve these"]:
            result = runtime.handle(case)
            print(f"[ok={result.verified}] {case}")
            print(result.response[:2400])
            print()
            if not result.verified:
                raise SystemExit(f"Expected '{case}' to run read-only.")
            assert_contains(
                result.response,
                [
                    "Approval review",
                    "Approval #",
                    "run_shell_command",
                    "get_clipboard",
                    "enable_computer_control",
                    "create_reminder",
                    "Why gated",
                    "Safer check",
                    "approval readiness",
                    "approval packet",
                    "Approval proof chain",
                    "approval chain proof",
                    "verification receipt <approved run id from approval chain proof",
                    "approve approval",
                    "dismiss approval",
                    "Approval decision matrix",
                    "Approve only if",
                    "Dismiss if",
                    "Narrow first if",
                    "Before approving",
                    "explicit stop times",
                    "Computer-control approval checklist",
                    "computer readiness",
                    "computer task plan",
                    "one primitive step",
                ],
                case,
            )
            assert_operator_limits(result.tool_results[0].metadata, case)
            assert_pending_approvals_path_display(result.tool_results[0].metadata, case)
            metadata = result.tool_results[0].metadata
            if metadata.get("approval_review_handoff_ready") is not True:
                raise SystemExit(f"{case} missed ready approval review handoff: {metadata}")
            if metadata.get("approval_review_pending_count") != 4 or len(metadata.get("approval_review_items") or []) != 4:
                raise SystemExit(f"{case} missed structured approval review items: {metadata}")
            if metadata.get("approval_review_next_command_count") != len(metadata.get("approval_review_next_commands") or []):
                raise SystemExit(f"{case} approval review next-command count drifted: {metadata}")
            expected_first = {
                "approval_review_first_approval_id": 4,
                "approval_review_first_tool_name": "create_reminder",
                "approval_review_first_readiness_command": "approval readiness 4",
                "approval_review_first_last_look_command": "approval packet 4",
                "approval_review_first_proof_command": "approval chain proof 4",
                "approval_review_first_verification_command": "verification receipt <approved run id from approval chain proof 4>",
                "approval_review_first_approve_command": "approve approval 4",
                "approval_review_first_dismiss_command": "dismiss approval 4",
                "next_command": "approval readiness 4",
                "next_required_command": "approval readiness 4",
                "next_proof_command": "approval chain proof 4",
            }
            for key, expected in expected_first.items():
                if metadata.get(key) != expected:
                    raise SystemExit(f"{case} approval review field {key} drifted: {metadata}")
            handoff = metadata.get("approval_review_handoff") or {}
            if handoff.get("handoff_ready") is not True or handoff.get("approval_review_handoff_ready") is not True:
                raise SystemExit(f"{case} nested approval review handoff not ready: {metadata}")
            nested_expected = {
                "pending_count": 4,
                "first_approval_id": 4,
                "first_tool_name": "create_reminder",
                "first_readiness_command": "approval readiness 4",
                "first_last_look_command": "approval packet 4",
                "first_proof_command": "approval chain proof 4",
                "first_verification_command": "verification receipt <approved run id from approval chain proof 4>",
                "first_approve_command": "approve approval 4",
                "first_dismiss_command": "dismiss approval 4",
                "next_required_command": "approval readiness 4",
                "next_proof_command": "approval chain proof 4",
            }
            for key, expected in nested_expected.items():
                if handoff.get(key) != expected:
                    raise SystemExit(f"{case} nested approval review field {key} drifted: {metadata}")
            for key in ["approves_request", "dismisses_request", "authorizes_execution", "authorizes_completion_claim"]:
                if metadata.get(key) or handoff.get(key):
                    raise SystemExit(f"{case} approval review handoff should not grant authority through {key}: {metadata}")
            if not handoff.get("requires_manual_send"):
                raise SystemExit(f"{case} approval review handoff should require manual send: {metadata}")
            if handoff.get("next_commands") != metadata.get("approval_review_next_commands"):
                raise SystemExit(f"{case} approval review nested commands diverged: {metadata}")

        for case in ["approval detail 1", "inspect approval 2", "review pending approval 3"]:
            result = runtime.handle(case)
            print(f"[ok={result.verified}] {case}")
            print(result.response[:2400])
            print()
            if not result.verified:
                raise SystemExit(f"Expected '{case}' to run read-only.")
            assert_contains(
                result.response,
                [
                    "Approval detail #",
                    "Original request",
                    "Category",
                    "Why gated",
                    "Safer check",
                    "Rerun fingerprint",
                    "planned arg keys",
                    "Last-look checklist",
                    "explicit stop times",
                    "Approval decision matrix",
                    "explicit stop times",
                    "Approve only if",
                    "Dismiss if",
                    "Narrow first if",
                    "does not grant ongoing permission",
                    "Blocked output",
                    "Decision commands",
                    "approval readiness",
                    "approval packet",
                    "Approval proof chain",
                    "approval chain proof",
                    "verification receipt <approved run id from approval chain proof",
                    "approval reruns the exact original request once",
                ],
                case,
            )
            metadata = result.tool_results[0].metadata
            if (
                metadata.get("executes_tools")
                or metadata.get("queues_approval")
                or metadata.get("approves_request")
                or metadata.get("dismisses_request")
                or metadata.get("controls_computer")
                or metadata.get("reads_private_data")
                or metadata.get("writes_files")
            ):
                raise SystemExit(f"{case} should not act or queue approvals.")
            assert_operator_limits(metadata, case)
            assert_operator_limits(metadata, case)

        result = runtime.handle("approval detail 3")
        print(f"[ok={result.verified}] approval detail 3 computer checklist")
        print(result.response[:2400])
        print()
        if not result.verified:
            raise SystemExit("Expected computer approval detail to run read-only.")
        assert_contains(
            result.response,
            [
                "enable_computer_control",
                "Computer-control approval checklist",
                "computer readiness",
                "computer task plan",
                "one primitive step",
                "Stop if private messages",
            ],
            "computer approval detail",
        )

        latest_readiness = runtime.handle("approval readiness 4")
        if not latest_readiness.verified or latest_readiness.tool_results[0].metadata.get("approval_readiness_receipt_issued") is not True:
            raise SystemExit(f"Expected newest approval readiness to issue a receipt: {latest_readiness.tool_results}")

        for case in ["approval execution packet 1", "approval packet 3"]:
            result = runtime.handle(case)
            print(f"[blocked={not result.verified}] {case}")
            print(result.response[:2400])
            print()
            if result.verified or "needs a current readiness receipt" not in result.response:
                raise SystemExit(f"Expected older '{case}' to fail closed without readiness.")
            metadata = result.tool_results[0].metadata
            if metadata.get("approval_packet_viewed") is not False or metadata.get("approval_readiness_valid") is not False:
                raise SystemExit(f"{case} should not arm an older approval: {metadata}")

        for case in ["approval last look 4"]:
            result = runtime.handle(case)
            print(f"[ok={result.verified}] {case}")
            print(result.response[:2400])
            print()
            if not result.verified:
                raise SystemExit(f"Expected newest '{case}' to run read-only after readiness.")
            assert_contains(
                result.response,
                [
                    "Approval execution packet",
                    "What approval would do",
                    "Rerun exactly",
                    "Rerun fingerprint",
                    "planned arg keys",
                    "Last-look checklist",
                    "Approval decision matrix",
                    "Approve only if",
                    "Dismiss if",
                    "Narrow first if",
                    "does not grant ongoing permission",
                    "Approval readiness",
                    "Approval proof chain",
                    "approval chain proof",
                    "Decision commands",
                    "approve approval",
                    "dismiss approval",
                    "does not approve",
                ],
                case,
            )
            metadata = result.tool_results[0].metadata
            if (
                metadata.get("executes_tools")
                or metadata.get("queues_approval")
                or metadata.get("approves_request")
                or metadata.get("dismisses_request")
                or metadata.get("controls_computer")
                or metadata.get("reads_private_data")
                or metadata.get("writes_files")
            ):
                raise SystemExit(f"{case} should not act or queue approvals.")

        natural_last_look_cases = [
            ("show me the last look for approval 1", "1"),
            ("what would approval 3 do", "3"),
            ("preview what happens if I approve approval 4", "4"),
        ]
        for case, expected_id in natural_last_look_cases:
            result = runtime.handle(case)
            print(f"[ok={result.verified}] {case}")
            print(result.response[:1600])
            print()
            if result.tool_results[0].tool_name != "approval_execution_packet":
                raise SystemExit(f"Expected '{case}' to route to approval_execution_packet.")
            metadata = result.tool_results[0].metadata
            if str(metadata.get("approval_id")) != expected_id:
                raise SystemExit(f"Expected '{case}' to inspect approval #{expected_id}: {metadata}")
            if expected_id != "4":
                if result.verified or metadata.get("approval_packet_viewed") is not False:
                    raise SystemExit(f"Expected '{case}' to fail closed for an older approval: {metadata}")
                continue
            if not result.verified:
                raise SystemExit(f"Expected newest '{case}' to run read-only.")
            assert_contains(
                result.response,
                [
                    "Approval execution packet",
                    "Last-look checklist",
                    "Approval readiness",
                    "does not approve",
                    "Approval proof chain",
                    "explicit stop times",
                    "approval chain proof",
                ],
                case,
            )
            metadata = result.tool_results[0].metadata
            if not metadata.get("proof_chain_commands") or not str(metadata.get("next_proof_command") or "").startswith("approval chain proof"):
                raise SystemExit(f"{case} missed structured approval proof-chain metadata: {metadata}")
            if (
                metadata.get("executes_tools")
                or metadata.get("queues_approval")
                or metadata.get("approves_request")
                or metadata.get("dismisses_request")
            ):
                raise SystemExit(f"{case} should remain read-only.")
            assert_operator_limits(metadata, case)

        natural_readiness_cases = [
            ("is approval 1 ready to run", "1"),
            ("is it safe to approve approval 2", "2"),
            ("can I run approval 3 now", "3"),
            ("what should I check before approving approval 4", "4"),
            ("should I approve the latest approval", "4"),
            ("should I approve this approval", "4"),
            ("is it okay to approve approval 2", "2"),
            ("can I approve approval 2 safely", "2"),
            ("approval 2 safe?", "2"),
            ("check approval 2 before approving", "2"),
            ("review approval 2 before approving", "2"),
        ]
        for case, expected_id in natural_readiness_cases:
            result = runtime.handle(case)
            print(f"[ok={result.verified}] {case}")
            print(result.response[:1600])
            print()
            if not result.verified:
                raise SystemExit(f"Expected '{case}' to run read-only.")
            if result.tool_results[0].tool_name != "approval_readiness_packet":
                raise SystemExit(f"Expected '{case}' to route to approval_readiness_packet.")
            metadata = result.tool_results[0].metadata
            if str(metadata.get("approval_id")) != expected_id:
                raise SystemExit(f"Expected '{case}' to inspect approval #{expected_id}: {metadata}")
            assert_contains(
                result.response,
                [
                    "Approval readiness",
                    "Approval readiness verdict",
                    "next safe command",
                    "next required command after approval",
                    "Approval proof chain",
                    "explicit stop times",
                    "approve command stays disabled",
                ],
                case,
            )
            if "next proof command after approval" in result.response:
                raise SystemExit(f"{case} leaked old proof-first after-approval handoff label.")
            if not str(metadata.get("next_command") or "").strip():
                raise SystemExit(f"{case} missed next_command metadata: {metadata}")
            if not metadata.get("proof_chain_commands") or not str(metadata.get("next_proof_command") or "").startswith("approval chain proof"):
                raise SystemExit(f"{case} missed structured approval proof-chain metadata: {metadata}")
            if (
                metadata.get("executes_tools")
                or metadata.get("queues_approval")
                or metadata.get("approves_request")
                or metadata.get("dismisses_request")
            ):
                raise SystemExit(f"{case} should remain read-only.")
            assert_operator_limits(metadata, case)

        result = runtime.handle("approval packet 1")
        if result.verified or result.tool_results[0].metadata.get("approval_packet_viewed") is not False:
            raise SystemExit("Older shell approval packet should remain unarmed without current readiness.")
        result = runtime.handle("approval packet 3")
        if result.verified or result.tool_results[0].metadata.get("approval_packet_viewed") is not False:
            raise SystemExit("Older computer-control approval packet should remain unarmed without current readiness.")

        result = runtime.handle("dismiss approval 2")
        print(f"[ok={result.verified}] dismiss approval 2")
        print(result.response[:1200])
        print()
        if not result.verified:
            raise SystemExit("Expected dismiss approval 2 to run.")

        result = runtime.handle("approval history")
        print(f"[ok={result.verified}] approval history")
        print(result.response[:2400])
        print()
        if not result.verified:
            raise SystemExit("Expected approval history to run read-only.")
        assert_contains(
            result.response,
            [
                "Approval history",
                "audit-only",
                "Approval #",
                "[pending]",
                "[dismissed]",
                "approval packet",
                "explicit stop times",
                "Boundary",
                "does not approve",
            ],
            "approval history",
        )
        metadata = result.tool_results[0].metadata
        if (
            metadata.get("executes_tools")
            or metadata.get("queues_approval")
            or metadata.get("approves_request")
            or metadata.get("dismisses_request")
            or metadata.get("controls_computer")
            or metadata.get("reads_private_data")
            or metadata.get("writes_files")
        ):
            raise SystemExit("Approval history should not act or queue approvals.")
        assert_operator_limits(metadata, "approval history")

        result = runtime.handle("save approval review")
        print(f"[ok={result.verified}] save approval review")
        print(result.response[:2400])
        print()
        if not result.verified:
            raise SystemExit("Expected save approval review to run.")
        assert_contains(
            result.response,
            [
                "Approval review saved",
                "Approval review",
                "run_shell_command",
                "enable_computer_control",
                "create_reminder",
                "Before approving",
                "Computer-control approval checklist",
            ],
            "save approval review",
        )
        if not approval_review_note.exists():
            raise SystemExit("Approval review note was not written to Obsidian.")
        metadata = result.tool_results[0].metadata
        if metadata.get("pending_approvals_path_display") != "Automations/Pending Approvals.md":
            raise SystemExit(f"save approval review missed safe pending approvals display path: {metadata}")
        if str(metadata.get("pending_approvals_path") or "").endswith("Jarvis/Automations/Pending Approvals.md") is not True:
            raise SystemExit(f"save approval review missed exact pending approvals path: {metadata}")
        note_text = approval_review_note.read_text(encoding="utf-8")
        assert_contains(
            note_text,
            [
                "# Approval Review",
                "Approval #",
                "Why gated",
                "Safer check",
                "approve approval",
                "dismiss approval",
                "Approval decision matrix",
                "Approval proof chain",
                "approval chain proof",
                "Computer-control approval checklist",
            ],
            "approval review note",
        )


if __name__ == "__main__":
    main()
