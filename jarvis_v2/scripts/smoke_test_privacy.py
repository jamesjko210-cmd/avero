from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from jarvis_v2.agent.planner import RuleBasedPlanner
from jarvis_v2.scripts.test_runtime import make_temp_runtime


def test_planner_routes_privacy_report_aliases() -> None:
    # Real gap found live 2026-07-09: "what data do you use" / "what data does
    # jarvis use" fell through to chat while sibling phrasings like "what data
    # can jarvis access" and "show privacy report" both worked.
    p = RuleBasedPlanner()
    for q in ("privacy report", "what data do you use", "what data does jarvis use"):
        actions = p.plan(q).actions
        if [a.tool_name for a in actions] != ["privacy_report"]:
            raise SystemExit(f"privacy_report route missed: {q!r} -> {[a.tool_name for a in actions]}")


def assert_contains(text: str, expected: list[str], label: str) -> None:
    missing = [item for item in expected if item not in text]
    if missing:
        raise SystemExit(f"{label} missing expected text: {missing}")


def assert_privacy_report_handles_malformed_pending_approvals() -> None:
    class ExplodingRow:
        def __getitem__(self, key: str) -> object:
            raise RuntimeError("secret privacy row /\x55sers/example/private/approval.sqlite")

        def __str__(self) -> str:
            raise RuntimeError("secret privacy string /\x55sers/example/private/contact")

    with TemporaryDirectory(prefix="jarvis-privacy-rows-") as temp:
        runtime = make_temp_runtime(Path(temp))
        original_pending = runtime.store.list_pending_approvals
        runtime.store.list_pending_approvals = lambda status="pending", limit=25: [
            ExplodingRow(),
            {
                "id": 88,
                "tool_name": "send_telegram /\x55sers/example/private/token",
                "user_input": "message /\x55sers/example/private/contact saying hello",
            },
        ]
        try:
            report = runtime.registry.get("privacy_report").handler({})
        finally:
            runtime.store.list_pending_approvals = original_pending

        if not report.ok:
            raise SystemExit(f"malformed-row privacy report failed: {report}")
        for leaked in [
            "/\x55sers/operator",
            "private/approval.sqlite",
            "secret privacy",
            "private/contact",
        ]:
            if leaked in report.output:
                raise SystemExit(f"malformed-row privacy report leaked {leaked}: {report.output}")
        if "pending approval row(s) hidden for safety" not in report.output:
            raise SystemExit(f"malformed-row privacy report missed hidden-row diagnostic: {report.output}")
        if "#88 send_telegram <local-path>" not in report.output:
            raise SystemExit(f"malformed-row privacy report missed redacted readable approval: {report.output}")
        metadata = report.metadata
        if metadata.get("pending_approvals") != 2:
            raise SystemExit(f"malformed-row privacy report should preserve total pending count: {metadata}")
        if metadata.get("readable_pending_approvals") != 1:
            raise SystemExit(f"malformed-row privacy report missed readable pending count: {metadata}")
        if metadata.get("unreadable_pending_approval_rows") != 1:
            raise SystemExit(f"malformed-row privacy report missed unreadable pending count: {metadata}")
        for key in [
            "calls_model",
            "executes_tools",
            "queues_approval",
            "authorizes_execution",
            "authorizes_completion_claim",
            "approval_granted",
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
                raise SystemExit(f"malformed-row privacy report unsafe metadata {key}: {metadata}")


def main() -> None:
    test_planner_routes_privacy_report_aliases()
    readme = Path(__file__).resolve().parents[2] / "README.md"
    readme_text = readme.read_text()
    privacy_line = next((line for line in readme_text.splitlines() if line.startswith("- Privacy report:")), "")
    if "not migrated yet" in privacy_line:
        raise SystemExit(f"README privacy report line used stale migration wording: {privacy_line}")
    for expected in ["narrow V2 personal connectors are active", "broader legacy scopes remain blocked"]:
        if expected not in privacy_line:
            raise SystemExit(f"README privacy report line missed {expected!r}: {privacy_line}")

    assert_privacy_report_handles_malformed_pending_approvals()

    with TemporaryDirectory(prefix="jarvis-privacy-") as temp:
        runtime = make_temp_runtime(Path(temp))
        setup_cases = [
            "run command python3 --version",
            "get clipboard",
            "enable computer control",
        ]
        for case in setup_cases:
            result = runtime.handle(case)
            print(f"[blocked={not result.verified}] {case}")
            print(result.response[:1200])
            print()
            if result.verified:
                raise SystemExit(f"Expected '{case}' to be blocked without approval.")

        cases = [
            "privacy report",
            "what data can Jarvis access",
        ]
        for case in cases:
            result = runtime.handle(case)
            status = "ok" if result.verified else "blocked"
            print(f"[{status}] {case}")
            print(result.response[:2200])
            print()
            if not result.verified:
                raise SystemExit(f"Expected '{case}' to run as read-only.")
            assert_contains(
                result.response,
                [
                    "Jarvis privacy boundary report",
                    "Default rule",
                    "Approval-gated privacy surfaces",
                    "Clipboard reads",
                    "Risk-gated tools",
                    "Pending privacy approvals",
                    "Current connector privacy posture",
                    "calendar: active through narrow V2 commands",
                    "email: active through narrow V2 commands",
                    "messages: active through narrow V2 commands",
                    "contacts: active through narrow V2 commands",
                    "Still blocked legacy scope",
                    "run_shell_command",
                    "get_clipboard",
                    "enable_computer_control",
                    "broad calendar/email/messages account access",
                    "Jarvis-owned Obsidian folder: Obsidian root configured",
                    "watched metadata path: watched directory #1 configured",
                ],
                case,
            )
            if "Not migrated yet" in result.response:
                raise SystemExit(f"privacy report used stale migration wording: {result.response}")
            for leaked in [str(Path(temp)), "/private/", "/var/folders/", "/\x55sers/"]:
                if leaked in result.response:
                    raise SystemExit(f"privacy report leaked local path fragment {leaked!r}: {result.response}")
            metadata = result.tool_results[0].metadata
            if metadata.get("pending_approvals", 0) < 3 or metadata.get("risk_gated_tools", 0) < 3:
                raise SystemExit("privacy report metadata missed pending approvals or risk-gated tool counts.")
            expected_connector_names = ["calendar", "email", "messages", "contacts"]
            if metadata.get("gated_v2_connector_names") != expected_connector_names:
                raise SystemExit(f"privacy report missed active gated V2 connector names: {metadata}")
            if metadata.get("gated_v2_connector_count") != len(expected_connector_names):
                raise SystemExit(f"privacy report missed active gated V2 connector count: {metadata}")
            if metadata.get("still_blocked_legacy_scope_count") != len(metadata.get("still_blocked_legacy_scope") or []):
                raise SystemExit(f"privacy report legacy blocked scope count mismatch: {metadata}")
            if metadata.get("microphone_input_active") is not False:
                raise SystemExit(f"privacy report should keep microphone input inactive: {metadata}")
            for key in [
                "calls_model",
                "executes_tools",
                "queues_approval",
                "authorizes_execution",
                "authorizes_completion_claim",
                "approval_granted",
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
                    raise SystemExit(f"privacy report unsafe metadata {key}: {metadata}")


if __name__ == "__main__":
    main()
