from __future__ import annotations

import json
import tempfile
from enum import IntEnum
from pathlib import Path

from jarvis_v2.agent.executor import Executor
from jarvis_v2.agent.runtime import MAX_RUNTIME_OUTPUT_PREVIEW_CHARS, JarvisRuntime
from jarvis_v2.agent.types import Plan, PlannedAction, RiskLevel, ToolResult
from jarvis_v2.agent.verifier import MAX_VERIFICATION_FAILURE_CHARS, Verifier
from jarvis_v2.scripts.test_runtime import make_temp_runtime
from jarvis_v2.tools import contacts_connector, contacts_fuzzy
from jarvis_v2.tools.permissions import PermissionPolicy
from jarvis_v2.tools.registry import Tool, ToolRegistry


class HostileVerifierValue:
    def __init__(self, marker: str) -> None:
        self.marker = marker

    def __bool__(self) -> bool:
        raise RuntimeError(self.marker)

    def __str__(self) -> str:
        raise RuntimeError(self.marker)

    def __repr__(self) -> str:
        return "<hostile-verifier-value>"


class HostileExecutorValue:
    def __init__(self, marker: str) -> None:
        self.marker = marker

    def __bool__(self) -> bool:
        raise RuntimeError(self.marker)

    def __str__(self) -> str:
        raise RuntimeError(self.marker)

    def __repr__(self) -> str:
        return "<hostile-executor-value>"


class MalformedOutput(str):
    pass


def assert_no_local_path(text: str, label: str) -> None:
    if any(fragment in text for fragment in ["/\x55sers/", "Desktop/Claude code", "/private/", "/var/folders/", "/tmp/"]) or "\n" in text:
        raise SystemExit(f"{label} leaked a local path or newline: {text}")


def assert_contact_recovery_guidance(
    result: ToolResult,
    *,
    action: str,
    commands: list[str],
    label: str,
) -> None:
    expected = {"version": 1, "action": action, "commands": commands}
    if result.metadata.get("recovery_guidance") != expected:
        raise SystemExit(f"{label} recovery declaration drifted: {result.metadata}")
    if (
        result.metadata.get("next_command") != commands[0]
        or result.metadata.get("recovery_commands") != commands
    ):
        raise SystemExit(f"{label} recovery aliases drifted: {result.metadata}")
    if action not in result.output or any(command not in result.output for command in commands):
        raise SystemExit(f"{label} recovery is not visible to the user: {result.output}")
    for key in (
        "handler_invoked",
        "authorizes_retry",
        "authorizes_execution",
        "authorizes_completion_claim",
        "approval_granted",
    ):
        if result.metadata.get(key) is not False:
            raise SystemExit(f"{label} recovery granted authority through {key}: {result.metadata}")
    assert_no_local_path(result.output, f"{label} output")


def main() -> None:
    class ForeignRisk(IntEnum):
        LOOKS_SAFE = 0

    invalid_risks = (-1, 0, True, ForeignRisk.LOOKS_SAFE, object())
    for index, invalid_risk in enumerate(invalid_risks):
        invalid_registry = ToolRegistry()
        calls: list[dict] = []
        try:
            invalid_registry.register(
                Tool(
                    f"invalid_risk_{index}",
                    "Must never register.",
                    invalid_risk,  # type: ignore[arg-type]
                    lambda args: calls.append(dict(args)) or ToolResult("invalid", True, "ran"),
                    "test",
                )
            )
        except ValueError:
            pass
        else:
            raise SystemExit(f"Registry accepted malformed risk value: {invalid_risk!r}")
        if calls:
            raise SystemExit("Malformed risk registration invoked a handler")
    valid_registry = ToolRegistry()
    for risk in RiskLevel:
        valid_registry.register(
            Tool(f"valid_risk_{risk.name.lower()}", "Valid risk.", risk, lambda _: ToolResult("valid", True, "ok"), "test")
        )
    try:
        PermissionPolicy(max_auto_risk=0)  # type: ignore[arg-type]
    except ValueError:
        pass
    else:
        raise SystemExit("Permission policy accepted a plain integer risk ceiling")

    registry = ToolRegistry()

    def exploding_tool(_: dict) -> ToolResult:
        raise RuntimeError("boom from /\x55sers/example/private")

    approved_send_calls: list[tuple[str, dict]] = []
    malformed_metadata_calls: list[dict] = []
    malformed_output_calls: list[dict] = []

    def send_imessage_tool(args: dict) -> ToolResult:
        approved_send_calls.append(("send_imessage", dict(args)))
        return ToolResult("send_imessage", True, "sent")

    def send_email_tool(args: dict) -> ToolResult:
        approved_send_calls.append(("send_email", dict(args)))
        return ToolResult("send_email", True, "sent")

    def call_contact_tool(args: dict) -> ToolResult:
        approved_send_calls.append(("call_contact", dict(args)))
        return ToolResult("call_contact", True, "called")

    def malformed_metadata_tool(args: dict) -> ToolResult:
        malformed_metadata_calls.append(dict(args))
        return ToolResult("malformed_metadata_tool", True, "private side effect may have happened", None)  # type: ignore[arg-type]

    def malformed_output_tool(args: dict) -> ToolResult:
        malformed_output_calls.append(dict(args))
        return ToolResult(
            "malformed_output_tool",
            True,
            MalformedOutput("MALFORMED_OUTPUT_SECRET /\x55sers/example/private/output.txt"),
        )

    registry.register(Tool("explode", "Raise a controlled test error.", RiskLevel.READ_ONLY, exploding_tool, "test"))
    registry.register(Tool("risky", "Require approval.", RiskLevel.HIGH_RISK, lambda _: ToolResult("risky", True, "ran"), "test"))
    registry.register(Tool("send_imessage", "Send an iMessage.", RiskLevel.HIGH_RISK, send_imessage_tool, "test"))
    registry.register(Tool("send_email", "Send an email.", RiskLevel.HIGH_RISK, send_email_tool, "test"))
    registry.register(Tool("call_contact", "Call a contact.", RiskLevel.HIGH_RISK, call_contact_tool, "test"))
    registry.register(
        Tool(
            "malformed_metadata_tool",
            "Return malformed metadata after invocation.",
            RiskLevel.HIGH_RISK,
            malformed_metadata_tool,
            "test",
        )
    )
    registry.register(
        Tool(
            "malformed_output_tool",
            "Return malformed output after invocation.",
            RiskLevel.READ_ONLY,
            malformed_output_tool,
            "test",
        )
    )
    registry.register(
        Tool(
            "expected_tool",
            "Return a deliberately misbound receipt.",
            RiskLevel.READ_ONLY,
            lambda _: ToolResult("wrong_tool", True, "wrong receipt"),
            "test",
        )
    )
    registry.register(
        Tool(
            "malformed_ok_tool",
            "Return a deliberately malformed success flag.",
            RiskLevel.READ_ONLY,
            lambda _: ToolResult("malformed_ok_tool", "false", "private false success SHOULD NOT APPEAR"),
            "test",
        )
    )
    registry.register(
        Tool(
            "forged_preexecution_tool",
            "Try to forge a pre-execution receipt after the handler runs.",
            RiskLevel.READ_ONLY,
            lambda _: ToolResult(
                "forged_preexecution_tool",
                False,
                "handler returned a controlled failure",
                {
                    "executed_handler": False,
                    "failure_kind": "contact_resolution_not_found",
                },
            ),
            "test",
        )
    )

    executor = Executor(registry, PermissionPolicy())

    malformed_metadata = executor.execute(
        PlannedAction("malformed_metadata_tool", {"value": "side"}, "exercise malformed metadata truth"),
        approved=True,
    )
    if malformed_metadata_calls != [{"value": "side"}] or malformed_metadata.ok is not False:
        raise SystemExit(f"Malformed metadata fixture did not run exactly once: {malformed_metadata_calls}")
    if malformed_metadata.metadata.get("failure_kind") != "tool_result_metadata_malformed":
        raise SystemExit(f"Malformed metadata was mislabeled as a handler failure: {malformed_metadata}")
    for key, expected in {
        "handler_invoked": True,
        "executed_handler": True,
        "side_effect_possible": True,
        "execution_outcome_unknown": True,
        "outcome_known": False,
        "retry_safe": False,
        "authorizes_retry": False,
        "authorizes_execution": False,
        "authorizes_completion_claim": False,
    }.items():
        if malformed_metadata.metadata.get(key) is not expected:
            raise SystemExit(f"Malformed metadata receipt drifted at {key}: {malformed_metadata.metadata}")
    if "private side effect" in malformed_metadata.output or "Tool failed while running" in malformed_metadata.output:
        raise SystemExit(f"Malformed metadata result leaked output or claimed handler failure: {malformed_metadata.output}")

    malformed_ok = executor.execute(
        PlannedAction("malformed_ok_tool", {"mode": "test"}, "exercise exact boolean receipt truth")
    )
    if malformed_ok.ok is not False or malformed_ok.metadata.get("failure_kind") != "tool_result_ok_malformed":
        raise SystemExit(f"Executor accepted a malformed success flag: {malformed_ok}")
    if malformed_ok.metadata.get("executed_handler") is not True or malformed_ok.metadata.get("handler_invoked") is not True:
        raise SystemExit(f"Malformed success receipt hid handler execution: {malformed_ok.metadata}")
    if "SHOULD NOT APPEAR" in malformed_ok.output or "private false success" in malformed_ok.output:
        raise SystemExit(f"Malformed success receipt leaked handler output: {malformed_ok.output}")

    malformed_output = executor.execute(
        PlannedAction("malformed_output_tool", {"mode": "side"}, "exercise exact string output truth")
    )
    if malformed_output_calls != [{"mode": "side"}] or malformed_output.ok is not False:
        raise SystemExit(f"Malformed output fixture did not run exactly once: {malformed_output_calls}")
    if malformed_output.metadata.get("failure_kind") != "tool_result_output_malformed":
        raise SystemExit(f"Malformed output was mislabeled as a handler failure: {malformed_output}")
    for key, expected in {
        "handler_invoked": True,
        "executed_handler": True,
        "side_effect_possible": True,
        "execution_outcome_unknown": True,
        "outcome_known": False,
        "retry_safe": False,
        "retry_requires_recovery_review": True,
        "authorizes_retry": False,
        "authorizes_execution": False,
        "authorizes_completion_claim": False,
    }.items():
        if malformed_output.metadata.get(key) is not expected:
            raise SystemExit(f"Malformed output receipt drifted at {key}: {malformed_output.metadata}")
    if "MALFORMED_OUTPUT_SECRET" in malformed_output.output or "/\x55sers/" in malformed_output.output:
        raise SystemExit(f"Malformed output result leaked the handler payload: {malformed_output.output}")
    if malformed_output.metadata.get("recovery_commands") != ["recent tool runs", "execution recovery"]:
        raise SystemExit(f"Malformed output receipt should require recovery review: {malformed_output.metadata}")

    forged_preexecution = executor.execute(
        PlannedAction(
            "forged_preexecution_tool",
            {},
            "prove executor ownership of handler-invocation truth",
        )
    )
    if (
        forged_preexecution.ok
        or forged_preexecution.metadata.get("handler_invoked") is not True
        or forged_preexecution.metadata.get("executed_handler") is not True
    ):
        raise SystemExit(
            "An invoked handler forged a pre-execution receipt: "
            f"{forged_preexecution.metadata}"
        )

    with tempfile.TemporaryDirectory(prefix="jarvis-malformed-output-audit-") as tmp:
        audit_runtime = make_temp_runtime(Path(tmp))
        audit_runtime._log_tool_runs([malformed_output], approved=False)
        run_id = malformed_output.metadata.get("logged_tool_run_id")
        stored_run = audit_runtime.store.get_tool_run(run_id) if type(run_id) is int else None
        if stored_run is None:
            raise SystemExit(f"Malformed output receipt was not audit logged: {malformed_output.metadata}")
        stored_metadata = json.loads(stored_run["metadata"])
        if "MALFORMED_OUTPUT_SECRET" in stored_run["output"] or "MALFORMED_OUTPUT_SECRET" in json.dumps(stored_metadata):
            raise SystemExit(f"Malformed output payload leaked into the temporary runtime audit: {dict(stored_run)}")
        for key, expected in {
            "handler_invoked": True,
            "executed_handler": True,
            "side_effect_possible": True,
            "execution_outcome_unknown": True,
            "outcome_known": False,
            "retry_safe": False,
        }.items():
            if stored_metadata.get(key) is not expected:
                raise SystemExit(f"Malformed output audit truth drifted at {key}: {stored_metadata}")

    misbound = executor.execute(PlannedAction("expected_tool", {"mode": "test"}, "exercise receipt binding"))
    if misbound.ok or misbound.tool_name != "expected_tool":
        raise SystemExit(f"Executor should fail a handler receipt bound to the wrong tool: {misbound}")
    if misbound.metadata.get("failure_kind") != "tool_result_binding_mismatch":
        raise SystemExit(f"Executor should expose a stable receipt-binding failure: {misbound.metadata}")
    if misbound.metadata.get("executed_handler") is not True:
        raise SystemExit(f"Misbound handler still ran and should be recorded truthfully: {misbound.metadata}")
    if misbound.metadata.get("reported_tool_name") != "wrong_tool":
        raise SystemExit(f"Executor should preserve a bounded reported tool name: {misbound.metadata}")
    for key in ["authorizes_retry", "authorizes_execution", "authorizes_completion_claim"]:
        if misbound.metadata.get(key) is not False:
            raise SystemExit(f"Receipt-binding failure should not grant authority via {key}: {misbound.metadata}")

    missing = executor.execute(PlannedAction("missing_tool", {"x": 1}, "exercise missing tool metadata"))
    if missing.ok or missing.metadata.get("failure_kind") != "unknown_tool":
        raise SystemExit(f"Missing tool should expose unknown_tool metadata: {missing}")
    if missing.metadata.get("executed_handler") is not False or missing.metadata.get("requires_confirmation") is not False:
        raise SystemExit(f"Missing tool should not look executed or approval-held: {missing.metadata}")
    if missing.metadata.get("planned_args") != {"x": 1}:
        raise SystemExit(f"Missing tool should preserve planned args for recovery: {missing.metadata}")
    if missing.metadata.get("planned_args_display") != {"x": 1}:
        raise SystemExit(f"Missing tool should expose safe planned args display metadata: {missing.metadata}")
    if missing.metadata.get("requested_tool") != "missing_tool" or missing.metadata.get("exception_type") != "KeyError":
        raise SystemExit(f"Missing tool should expose bounded diagnostic metadata: {missing.metadata}")
    if "list tools" not in missing.output or "before retry" not in missing.output:
        raise SystemExit(f"Missing tool should name the registry recovery command: {missing.output}")
    if missing.metadata.get("next_command") != "list tools" or missing.metadata.get("recovery_commands") != ["list tools"]:
        raise SystemExit(f"Missing tool should expose an exact recovery command: {missing.metadata}")
    if missing.metadata.get("retry_requires_registry_fix") is not True:
        raise SystemExit(f"Missing tool should block retry until registry repair: {missing.metadata}")
    for key in ["authorizes_retry", "authorizes_execution", "authorizes_completion_claim"]:
        if missing.metadata.get(key) is not False:
            raise SystemExit(f"Missing tool recovery metadata should not grant authority via {key}: {missing.metadata}")
    if "Unknown tool" in missing.output or "missing_tool" in missing.output or "KeyError" in missing.output:
        raise SystemExit(f"Missing tool should not leak raw registry exception text: {missing.output}")
    long_name = "missing_" + ("x" * 220)
    long_missing = executor.execute(PlannedAction(long_name, {}, "exercise bounded missing tool metadata"))
    if long_missing.ok or long_missing.metadata.get("failure_kind") != "unknown_tool":
        raise SystemExit(f"Long missing tool should fail as unknown_tool: {long_missing}")
    if long_missing.metadata.get("requested_tool") != ("missing_" + "x" * 151 + "…"):
        raise SystemExit(f"Missing tool should bound requested_tool metadata: {long_missing.metadata}")
    path_missing = executor.execute(
        PlannedAction(
            "missing_path_tool",
            {
                "path": "/\x55sers/example/Desktop/Claude code/AI agents/jarvis-v2/secret.txt",
                "nested": {"tmp": "/private/tmp/jarvis-missing-secret.txt"},
            },
            "exercise missing tool path display metadata",
        )
    )
    if path_missing.ok or path_missing.metadata.get("failure_kind") != "unknown_tool":
        raise SystemExit(f"Path-shaped missing tool should fail as unknown_tool: {path_missing}")
    missing_display = path_missing.metadata.get("planned_args_display")
    if missing_display != {"path": "<local-path>", "nested": {"tmp": "<local-path>"}}:
        raise SystemExit(f"Missing tool should scrub planned args display metadata: {path_missing.metadata}")
    assert_no_local_path(str(missing_display), "missing tool planned args display")
    if path_missing.metadata.get("planned_args", {}).get("path") != "/\x55sers/example/Desktop/Claude code/AI agents/jarvis-v2/secret.txt":
        raise SystemExit("Missing tool should preserve exact planned args for recovery.")

    hostile_executor_marker = "EXECUTOR_SHOULD_NOT_LEAK /\x55sers/example/private/executor-secret.txt"
    hostile_executor_value = HostileExecutorValue(hostile_executor_marker)
    hostile_missing = executor.execute(
        PlannedAction(
            "missing_tool",
            {
                hostile_executor_value: hostile_executor_value,
                "nested": {hostile_executor_value: [hostile_executor_value]},
                "path": "/\x55sers/example/Desktop/Claude code/AI agents/jarvis-v2/hostile-secret.txt",
            },
            "exercise hostile missing tool display metadata",
        )
    )
    if hostile_missing.ok or hostile_missing.metadata.get("failure_kind") != "unknown_tool":
        raise SystemExit(f"Hostile missing tool should fail as unknown_tool: {hostile_missing}")
    hostile_missing_display = hostile_missing.metadata.get("planned_args_display")
    if hostile_missing_display != {
        "<unreadable>": "<unreadable>",
        "nested": {"<unreadable>": ["<unreadable>"]},
        "path": "<local-path>",
    }:
        raise SystemExit(f"Hostile missing tool should expose only safe display metadata: {hostile_missing.metadata}")
    hostile_display_text = str(hostile_missing_display)
    assert_no_local_path(hostile_display_text, "hostile missing tool planned args display")
    if "EXECUTOR_SHOULD_NOT_LEAK" in hostile_display_text or "executor-secret" in hostile_display_text:
        raise SystemExit(f"Hostile missing tool display leaked marker text: {hostile_display_text}")
    if hostile_missing.metadata.get("planned_args", {}).get(hostile_executor_value) is not hostile_executor_value:
        raise SystemExit("Hostile missing tool should preserve exact planned args for recovery.")

    held = executor.execute(
        PlannedAction(
            "risky",
            {
                "target": "screen",
                "path": "/var/folders/zc/approval-hold-secret.log",
                "items": ["/tmp/approval-hold-secret.log"],
            },
            "exercise approval hold metadata",
        )
    )
    if held.ok or held.metadata.get("failure_kind") != "approval_required":
        raise SystemExit(f"Approval hold should expose approval_required metadata: {held}")
    if held.metadata.get("requires_confirmation") is not True or held.metadata.get("executed_handler") is not False:
        raise SystemExit(f"Approval hold should stop before handler execution: {held.metadata}")
    if held.metadata.get("risk_level") != "HIGH_RISK" or held.metadata.get("risk_value") != int(RiskLevel.HIGH_RISK):
        raise SystemExit(f"Approval hold should expose tool risk metadata: {held.metadata}")
    if held.metadata.get("toolset") != "test":
        raise SystemExit(f"Approval hold should expose toolset metadata: {held.metadata}")
    held_display = held.metadata.get("planned_args_display")
    if held_display != {"target": "screen", "path": "<local-path>", "items": ["<local-path>"]}:
        raise SystemExit(f"Approval hold should scrub planned args display metadata: {held.metadata}")
    assert_no_local_path(str(held_display), "approval hold planned args display")
    if held.metadata.get("planned_args", {}).get("path") != "/var/folders/zc/approval-hold-secret.log":
        raise SystemExit("Approval hold should preserve exact planned args for pending approval matching.")

    hostile_held = executor.execute(
        PlannedAction(
            "risky",
            {
                hostile_executor_value: hostile_executor_value,
                "items": [hostile_executor_value, "/tmp/hostile-approval-secret.log"],
            },
            "exercise hostile approval hold display metadata",
        )
    )
    if hostile_held.ok or hostile_held.metadata.get("failure_kind") != "approval_required":
        raise SystemExit(f"Hostile approval hold should require approval: {hostile_held}")
    hostile_held_display = hostile_held.metadata.get("planned_args_display")
    if hostile_held_display != {"<unreadable>": "<unreadable>", "items": ["<unreadable>", "<local-path>"]}:
        raise SystemExit(f"Hostile approval hold should expose only safe display metadata: {hostile_held.metadata}")
    hostile_held_text = str(hostile_held_display)
    assert_no_local_path(hostile_held_text, "hostile approval hold planned args display")
    if "EXECUTOR_SHOULD_NOT_LEAK" in hostile_held_text or "executor-secret" in hostile_held_text:
        raise SystemExit(f"Hostile approval hold display leaked marker text: {hostile_held_text}")

    original_resolve_contact = contacts_connector.resolve_contact
    original_looks_like_handle = contacts_connector.looks_like_handle
    original_fuzzy_contacts = contacts_fuzzy._run_all_contacts_query
    try:
        contacts_connector.resolve_contact = lambda query: [  # type: ignore[assignment]
            contacts_connector.ContactMatch(name="Fixture Example", phone="+821012345678", email="fixture@example.com")
        ]
        resolved_hold = executor.execute(PlannedAction("send_imessage", {"to": "fixture", "message": "hi"}, "test send resolution"))
        if resolved_hold.ok or resolved_hold.metadata.get("failure_kind") != "approval_required":
            raise SystemExit(f"Resolved send should still require approval: {resolved_hold}")
        if resolved_hold.metadata.get("planned_args", {}).get("to") != "+821012345678":
            raise SystemExit(f"Resolved send should queue the handle, not the raw name: {resolved_hold.metadata}")
        if resolved_hold.metadata.get("contact_resolution_status") != "resolved":
            raise SystemExit(f"Resolved send should expose contact resolution metadata: {resolved_hold.metadata}")
        if "Fixture Example" not in resolved_hold.output or "+821012345678" not in resolved_hold.output:
            raise SystemExit(f"Resolved approval output should name the resolved recipient: {resolved_hold.output}")

        resolved_email = executor.execute(PlannedAction("send_email", {"to": "fixture", "subject": "Hi", "body": "Hello"}, "test email resolution"))
        if resolved_email.metadata.get("planned_args", {}).get("to") != "fixture@example.com":
            raise SystemExit(f"Email sends should prefer an email handle before approval: {resolved_email.metadata}")

        resolved_call = executor.execute(
            PlannedAction(
                "call_contact",
                {"to": "fixture", "mode": "audio"},
                "test call resolution",
            )
        )
        if resolved_call.ok or resolved_call.metadata.get("failure_kind") != "approval_required":
            raise SystemExit(f"Resolved call should still require approval: {resolved_call}")
        if resolved_call.metadata.get("planned_args") != {
            "to": "+821012345678",
            "mode": "audio",
        }:
            raise SystemExit(f"Resolved call approval should bind the exact handle and mode: {resolved_call.metadata}")
        if approved_send_calls:
            raise SystemExit(f"Pre-approval call resolution invoked the call handler: {approved_send_calls}")

        contacts_connector.resolve_contact = lambda query: [  # type: ignore[assignment]
            contacts_connector.ContactMatch(
                name="Email Only Fixture",
                email="email.only@example.com",
            )
        ]
        email_only_phone = executor.execute(
            PlannedAction(
                "call_contact",
                {"to": "email only fixture", "mode": "phone"},
                "test email-only phone resolution",
            )
        )
        if (
            email_only_phone.ok
            or email_only_phone.metadata.get("failure_kind")
            != "contact_resolution_missing_phone_handle"
            or email_only_phone.metadata.get("requires_confirmation") is not False
            or approved_send_calls
        ):
            raise SystemExit(
                f"Email-only contact must not queue or place a phone call: {email_only_phone}"
            )
        assert_contact_recovery_guidance(
            email_only_phone,
            action=(
                "Add a phone number, run `find contact <name>`, then submit a fresh phone-call "
                "request; or request FaceTime audio/video."
            ),
            commands=["find contact <name>"],
            label="email-only phone call",
        )

        email_only_audio = executor.execute(
            PlannedAction(
                "call_contact",
                {"to": "email only fixture", "mode": "audio"},
                "test email-only FaceTime audio resolution",
            )
        )
        if email_only_audio.metadata.get("planned_args") != {
            "to": "email.only@example.com",
            "mode": "audio",
        }:
            raise SystemExit(
                f"FaceTime audio should bind an email-only contact: {email_only_audio.metadata}"
            )

        direct_email_phone = executor.execute(
            PlannedAction(
                "call_contact",
                {"to": "email.only@example.com", "mode": "phone"},
                "test direct email phone refusal",
            )
        )
        if (
            direct_email_phone.ok
            or direct_email_phone.metadata.get("failure_kind")
            != "contact_resolution_invalid_phone_handle"
            or direct_email_phone.metadata.get("requires_confirmation") is not False
            or approved_send_calls
        ):
            raise SystemExit(
                f"Direct email phone call must stop before approval: {direct_email_phone}"
            )

        approved_email_phone = executor.execute(
            PlannedAction(
                "call_contact",
                {"to": "email.only@example.com", "mode": "phone"},
                "test approved legacy email phone refusal",
            ),
            approved=True,
        )
        if (
            approved_email_phone.ok
            or approved_email_phone.metadata.get("failure_kind")
            != "contact_resolution_approved_invalid_phone_handle"
            or approved_email_phone.metadata.get("executed_handler") is not False
            or approved_send_calls
        ):
            raise SystemExit(
                f"Approved email phone call must fail before handler execution: {approved_email_phone}"
            )

        contacts_connector.resolve_contact = lambda query: [  # type: ignore[assignment]
            contacts_connector.ContactMatch(name="Email Fixture", phone="+821012300001", email="email.fixture@example.com"),
            contacts_connector.ContactMatch(name="Work Fixture", phone="+821012300002", email="work.fixture@example.com"),
        ]
        ambiguous_email = executor.execute(PlannedAction("send_email", {"to": "fixture", "subject": "Hi", "body": "Hello"}, "test ambiguous email"))
        email_choices = ambiguous_email.metadata.get("contact_resolution_matches")
        if ambiguous_email.metadata.get("failure_kind") != "contact_resolution_ambiguous":
            raise SystemExit(f"Ambiguous email should expose contact_resolution_ambiguous: {ambiguous_email.metadata}")
        if email_choices != [
            {"name": "Email Fixture", "handle": "email.fixture@example.com"},
            {"name": "Work Fixture", "handle": "work.fixture@example.com"},
        ]:
            raise SystemExit(f"Ambiguous email choices should preview email handles, not phone handles: {ambiguous_email.metadata}")

        contacts_connector.resolve_contact = lambda query: [  # type: ignore[assignment]
            contacts_connector.ContactMatch(name="Phone Only Fixture", phone="+821012300003")
        ]
        phone_only_email = executor.execute(PlannedAction("send_email", {"to": "fixture", "subject": "Hi", "body": "Hello"}, "test phone-only email"))
        if phone_only_email.metadata.get("failure_kind") != "contact_resolution_missing_handle":
            raise SystemExit(f"Phone-only email contact should fail before approval: {phone_only_email.metadata}")
        if phone_only_email.metadata.get("contact_resolution_matches") != [{"name": "Phone Only Fixture", "handle": ""}]:
            raise SystemExit(f"Phone-only email refusal should not preview phone number as email handle: {phone_only_email.metadata}")
        assert_contact_recovery_guidance(
            phone_only_email,
            action=(
                "Add the required phone number or email address in Contacts, then run "
                "`find contact <name>` before trying the request again."
            ),
            commands=["find contact <name>"],
            label="missing contact handle",
        )

        contacts_connector.resolve_contact = lambda query: [  # type: ignore[assignment]
            contacts_connector.ContactMatch(name="Fixture Example", phone="+821012345678"),
            contacts_connector.ContactMatch(name="Fixture Kim", phone="+821099999999"),
        ]
        ambiguous = executor.execute(PlannedAction("send_imessage", {"to": "fixture", "message": "hi"}, "test ambiguous send"))
        if ambiguous.ok or ambiguous.metadata.get("requires_confirmation") is not False:
            raise SystemExit(f"Ambiguous send should not queue approval: {ambiguous}")
        if ambiguous.metadata.get("failure_kind") != "contact_resolution_ambiguous":
            raise SystemExit(f"Ambiguous send should expose contact_resolution_ambiguous: {ambiguous.metadata}")
        if "Which one" not in ambiguous.output:
            raise SystemExit(f"Ambiguous send should ask for clarification: {ambiguous.output}")
        ambiguous_call = executor.execute(
            PlannedAction(
                "call_contact",
                {"to": "fixture", "mode": "audio"},
                "test ambiguous call",
            )
        )
        if (
            ambiguous_call.ok
            or ambiguous_call.metadata.get("failure_kind") != "contact_resolution_ambiguous"
            or ambiguous_call.metadata.get("requires_confirmation") is not False
            or approved_send_calls
        ):
            raise SystemExit(f"Ambiguous call should stop before approval and handler execution: {ambiguous_call}")

        contacts_connector.resolve_contact = lambda query: []  # type: ignore[assignment]
        contacts_fuzzy._run_all_contacts_query = lambda: "Fixture Example|||Fixturette Kim|||Fixture Secondary|||"  # type: ignore[assignment]
        not_found = executor.execute(PlannedAction("send_imessage", {"to": "fixture", "message": "hi"}, "test missing send"))
        if not_found.ok or not_found.metadata.get("requires_confirmation") is not False:
            raise SystemExit(f"Missing contact send should not queue approval: {not_found}")
        if not_found.metadata.get("failure_kind") != "contact_resolution_not_found":
            raise SystemExit(f"Missing contact send should expose contact_resolution_not_found: {not_found.metadata}")
        if "Did you mean Fixture Example" not in not_found.output or not_found.metadata.get("contact_resolution_suggestion_clause") == "":
            raise SystemExit(f"Missing contact send should include fuzzy suggestions before approval: {not_found.output} {not_found.metadata}")
        assert_contact_recovery_guidance(
            not_found,
            action=(
                "Correct or add the contact, then run `find contact <name>` before trying "
                "the request again."
            ),
            commands=["find contact <name>"],
            label="contact not found",
        )

        contacts_connector.resolve_contact = lambda query: (_ for _ in ()).throw(RuntimeError("contacts unavailable"))  # type: ignore[assignment]
        unavailable = executor.execute(PlannedAction("send_imessage", {"to": "fixture", "message": "hi"}, "test resolver failure"))
        if unavailable.ok or unavailable.metadata.get("requires_confirmation") is not False:
            raise SystemExit(f"Resolver failure should not queue approval: {unavailable}")
        if unavailable.metadata.get("failure_kind") != "contact_resolution_unavailable":
            raise SystemExit(f"Resolver failure should expose contact_resolution_unavailable: {unavailable.metadata}")
        if unavailable.metadata.get("executed_handler") is not False or unavailable.metadata.get("exception_type") != "RuntimeError":
            raise SystemExit(f"Resolver failure should be held at executor boundary: {unavailable.metadata}")
        assert_contact_recovery_guidance(
            unavailable,
            action=(
                "Run `setup check`, restore macOS Contacts permission, then run "
                "`find contact <name>` before trying the request again."
            ),
            commands=["setup check", "find contact <name>"],
            label="contact resolver unavailable",
        )
        if "contacts unavailable" in unavailable.output:
            raise SystemExit(f"Resolver failure leaked raw exception text: {unavailable.output}")
        unavailable_call = executor.execute(
            PlannedAction(
                "call_contact",
                {"to": "fixture", "mode": "audio"},
                "test unavailable call resolution",
            )
        )
        if (
            unavailable_call.ok
            or unavailable_call.metadata.get("failure_kind") != "contact_resolution_unavailable"
            or unavailable_call.metadata.get("requires_confirmation") is not False
            or approved_send_calls
        ):
            raise SystemExit(f"Unavailable Contacts should stop a call before approval: {unavailable_call}")

        contacts_connector.resolve_contact = lambda query: (_ for _ in ()).throw(RuntimeError("should not resolve handles"))  # type: ignore[assignment]
        direct_handle = executor.execute(PlannedAction("send_imessage", {"to": "+821012345678", "message": "hi"}, "test direct handle"))
        if direct_handle.metadata.get("planned_args", {}).get("to") != "+821012345678":
            raise SystemExit(f"Direct handles should pass through unchanged: {direct_handle.metadata}")
        if direct_handle.metadata.get("contact_resolution_status") != "already_handle":
            raise SystemExit(f"Direct handles should be marked already_handle: {direct_handle.metadata}")

        approved_send_calls.clear()
        stale_approved = executor.execute(PlannedAction("send_imessage", {"to": "fixture", "message": "hi"}, "test stale approved send"), approved=True)
        if stale_approved.ok or stale_approved.metadata.get("failure_kind") != "stale_send_approval_unresolved_recipient":
            raise SystemExit(f"Approved raw-name send should fail closed as stale/unresolved: {stale_approved}")
        if stale_approved.metadata.get("executed_handler") is not False or stale_approved.metadata.get("approval_rerun_blocked") is not True:
            raise SystemExit(f"Approved raw-name send should be blocked before handler execution: {stale_approved.metadata}")
        if approved_send_calls:
            raise SystemExit(f"Approved raw-name send should not execute handler: {approved_send_calls}")

        stale_email = executor.execute(PlannedAction("send_email", {"to": "fixture", "subject": "Hi", "body": "Hello"}, "test stale approved email"), approved=True)
        if stale_email.ok or stale_email.metadata.get("failure_kind") != "stale_send_approval_unresolved_recipient":
            raise SystemExit(f"Approved raw-name email should fail closed as stale/unresolved: {stale_email}")

        stale_call = executor.execute(
            PlannedAction(
                "call_contact",
                {"to": "fixture", "mode": "audio"},
                "test stale approved call",
            ),
            approved=True,
        )
        if (
            stale_call.ok
            or stale_call.metadata.get("failure_kind")
            != "stale_call_approval_unresolved_recipient"
            or stale_call.metadata.get("approval_rerun_block_reason")
            != "call_recipient_not_resolved_before_approval"
            or approved_send_calls
        ):
            raise SystemExit(f"Approved raw-name call should fail closed before dialing: {stale_call}")

        approved_direct = executor.execute(PlannedAction("send_imessage", {"to": "+821012345678", "message": "hi"}, "test approved direct send"), approved=True)
        if not approved_direct.ok or approved_send_calls != [("send_imessage", {"to": "+821012345678", "message": "hi"})]:
            raise SystemExit(f"Approved direct-handle send should still execute exactly once: {approved_direct} calls={approved_send_calls}")
        approved_call = executor.execute(
            PlannedAction(
                "call_contact",
                {"to": "+821012345678", "mode": "audio"},
                "test approved resolved call",
            ),
            approved=True,
        )
        if not approved_call.ok or approved_send_calls != [
            ("send_imessage", {"to": "+821012345678", "message": "hi"}),
            ("call_contact", {"to": "+821012345678", "mode": "audio"}),
        ]:
            raise SystemExit(f"Approved resolved-handle call should execute exactly once: {approved_call} calls={approved_send_calls}")
    finally:
        contacts_connector.resolve_contact = original_resolve_contact  # type: ignore[assignment]
        contacts_connector.looks_like_handle = original_looks_like_handle  # type: ignore[assignment]
        contacts_fuzzy._run_all_contacts_query = original_fuzzy_contacts  # type: ignore[assignment]

    original_resolve_contact = contacts_connector.resolve_contact
    original_fuzzy_contacts = contacts_fuzzy._run_all_contacts_query
    try:
        contacts_connector.resolve_contact = lambda query: [  # type: ignore[assignment]
            contacts_connector.ContactMatch(name="Fixture Example", phone="+821012345678")
        ]
        with tempfile.TemporaryDirectory(prefix="jarvis-resolved-approval-") as tmp:
            runtime_for_send = make_temp_runtime(Path(tmp))
            result = runtime_for_send.handle("text fixture saying hi")
            pending = runtime_for_send.store.list_pending_approvals(limit=10)
            if len(pending) != 1:
                raise SystemExit(f"Resolved natural send should queue exactly one approval: response={result.response!r} pending={len(pending)}")
            planned_args = json.loads(pending[0]["planned_args"])
            if planned_args.get("to") != "+821012345678" or planned_args.get("message") != "hi":
                raise SystemExit(f"Pending approval should store resolved handle and body: {planned_args}")
            if "Fixture Example" not in pending[0]["reason"] or "+821012345678" not in pending[0]["reason"]:
                raise SystemExit(f"Pending approval reason should show resolved name and handle: {pending[0]['reason']}")

        with tempfile.TemporaryDirectory(prefix="jarvis-resolved-call-approval-") as tmp:
            runtime_for_call = make_temp_runtime(Path(tmp))
            result = runtime_for_call.handle("FaceTime audio Fixture")
            pending = runtime_for_call.store.list_pending_approvals(limit=10)
            if len(pending) != 1:
                raise SystemExit(
                    "Resolved natural call should queue exactly one approval: "
                    f"response={result.response!r} pending={len(pending)}"
                )
            if pending[0]["tool_name"] != "call_contact":
                raise SystemExit(f"Natural FaceTime call queued the wrong tool: {pending[0]}")
            planned_args = json.loads(pending[0]["planned_args"])
            if planned_args != {"to": "+821012345678", "mode": "audio"}:
                raise SystemExit(f"Pending call approval should bind handle and mode: {planned_args}")
            if "Fixture Example" not in pending[0]["reason"] or "+821012345678" not in pending[0]["reason"]:
                raise SystemExit(f"Pending call approval reason should show the bound recipient: {pending[0]['reason']}")

        contacts_connector.resolve_contact = lambda query: [  # type: ignore[assignment]
            contacts_connector.ContactMatch(name="Fixture Example", phone="+821012345678"),
            contacts_connector.ContactMatch(name="Fixture Kim", phone="+821099999999"),
        ]
        with tempfile.TemporaryDirectory(prefix="jarvis-ambiguous-approval-") as tmp:
            runtime_for_send = make_temp_runtime(Path(tmp))
            result = runtime_for_send.handle("text fixture saying hi")
            pending = runtime_for_send.store.list_pending_approvals(limit=10)
            if pending:
                raise SystemExit(f"Ambiguous natural send should not queue approval: {pending}")
            if result.tool_results[0].metadata.get("failure_kind") != "contact_resolution_ambiguous":
                raise SystemExit(f"Ambiguous natural send should expose clarification metadata: {result.tool_results[0].metadata}")

        contacts_connector.resolve_contact = lambda query: []  # type: ignore[assignment]
        contacts_fuzzy._run_all_contacts_query = lambda: "Fixture Example|||Fixturette Kim|||Fixture Secondary|||"  # type: ignore[assignment]
        with tempfile.TemporaryDirectory(prefix="jarvis-missing-approval-") as tmp:
            runtime_for_send = make_temp_runtime(Path(tmp))
            result = runtime_for_send.handle("text fixture saying hi")
            pending = runtime_for_send.store.list_pending_approvals(limit=10)
            if pending:
                raise SystemExit(f"Missing-contact natural send should not queue approval: {pending}")
            if result.tool_results[0].metadata.get("failure_kind") != "contact_resolution_not_found":
                raise SystemExit(f"Missing-contact natural send should expose not-found metadata: {result.tool_results[0].metadata}")
            if "Did you mean Fixture Example" not in result.response:
                raise SystemExit(f"Missing-contact natural send should surface fuzzy suggestions before approval: {result.response}")
    finally:
        contacts_connector.resolve_contact = original_resolve_contact  # type: ignore[assignment]
        contacts_fuzzy._run_all_contacts_query = original_fuzzy_contacts  # type: ignore[assignment]

    errored = executor.execute(
        PlannedAction(
            "explode",
            {
                "mode": "test",
                "path": "/private/tmp/tool-error-secret.log",
                "nested": {"user_path": "/\x55sers/example/Desktop/Claude code/error-secret.txt"},
            },
            "exercise handler error metadata",
        )
    )
    if errored.ok or errored.metadata.get("failure_kind") != "tool_error":
        raise SystemExit(f"Tool exception should expose tool_error metadata: {errored}")
    if errored.metadata.get("executed_handler") is not True or errored.metadata.get("error_type") != "RuntimeError":
        raise SystemExit(f"Tool exception should expose handler/error metadata: {errored.metadata}")
    if errored.metadata.get("exception_type") != "RuntimeError":
        raise SystemExit(f"Tool exception should expose bounded exception metadata: {errored.metadata}")
    for expected in ["recent tool runs", "execution recovery", "latest failed run", "retry only after recovery review"]:
        if expected not in errored.output:
            raise SystemExit(f"Tool exception should name actionable recovery guidance {expected}: {errored.output}")
    expected_recovery_commands = ["recent tool runs", "execution recovery", "execution health report"]
    if errored.metadata.get("next_command") != expected_recovery_commands[0]:
        raise SystemExit(f"Tool exception should expose the first recovery command: {errored.metadata}")
    if errored.metadata.get("recovery_commands") != expected_recovery_commands:
        raise SystemExit(f"Tool exception should expose the ordered recovery commands: {errored.metadata}")
    if errored.metadata.get("retry_requires_recovery_review") is not True:
        raise SystemExit(f"Tool exception should block retry until recovery review: {errored.metadata}")
    for key in ["authorizes_retry", "authorizes_execution", "authorizes_completion_claim"]:
        if errored.metadata.get(key) is not False:
            raise SystemExit(f"Tool exception recovery metadata should not grant authority via {key}: {errored.metadata}")
    if "boom" in errored.output or "/\x55sers/operator" in errored.output or "Tool error:" in errored.output:
        raise SystemExit(f"Tool exception should not leak raw handler text: {errored.output}")
    if errored.metadata.get("planned_args", {}).get("mode") != "test":
        raise SystemExit(f"Tool exception should preserve planned args for recovery: {errored.metadata}")
    error_display = errored.metadata.get("planned_args_display")
    if error_display != {"mode": "test", "path": "<local-path>", "nested": {"user_path": "<local-path>"}}:
        raise SystemExit(f"Tool exception should scrub planned args display metadata: {errored.metadata}")
    assert_no_local_path(str(error_display), "tool error planned args display")

    verifier = Verifier()
    plan = Plan("exercise verifier failure preview", [PlannedAction("short_fail"), PlannedAction("long_fail")])
    short_failure = ToolResult("short_fail", False, "'run_shell_command' is risk level HIGH_RISK; explicit approval required.")
    raw_secret = "/\x55sers/example/Desktop/Claude code/private/secret.txt /var/folders/zc/verifier-secret.log /tmp/verifier-secret.log " + ("raw failure detail " * 40)
    long_failure = ToolResult("long_fail", False, raw_secret)
    verified, verification = verifier.verify(plan, [short_failure, long_failure])
    if verified:
        raise SystemExit("Verifier should fail when any result fails.")
    if "short_fail: 'run_shell_command' is risk level HIGH_RISK; explicit approval required." not in verification:
        raise SystemExit(f"Verifier should preserve short failure text: {verification}")
    if len(verification.split("long_fail: ", 1)[1]) > MAX_VERIFICATION_FAILURE_CHARS:
        raise SystemExit(f"Verifier long failure preview was not bounded: {verification}")
    if any(fragment in verification for fragment in ["/\x55sers/", "Desktop/Claude code", "/var/folders/", "/tmp/"]) or "\n" in verification:
        raise SystemExit(f"Verifier should normalize and truncate noisy failure text: {verification}")

    hostile_marker = "VERIFIER_SHOULD_NOT_LEAK /\x55sers/example/private/verifier-secret.txt"
    hostile_value = HostileVerifierValue(hostile_marker)
    verified, verification = verifier.verify(plan, [ToolResult(hostile_value, False, hostile_value)])
    if verified:
        raise SystemExit("Verifier should fail when hostile result fails.")
    if "<unknown tool>:" not in verification:
        raise SystemExit(f"Verifier should use a stable placeholder for hostile tool names: {verification}")
    if "VERIFIER_SHOULD_NOT_LEAK" in verification or "/\x55sers/" in verification or "verifier-secret" in verification:
        raise SystemExit(f"Verifier should not leak hostile marker or local path text: {verification}")

    successful_plan = Plan(
        "bind successful receipts to the plan",
        [PlannedAction("first_tool"), PlannedAction("second_tool")],
    )
    complete_results = [
        ToolResult("first_tool", True, "first done"),
        ToolResult("second_tool", True, "second done"),
    ]
    verified, verification = verifier.verify(successful_plan, complete_results)
    if not verified or verification != "All planned actions completed.":
        raise SystemExit(f"Verifier should accept an exact ordered receipt set: {verification}")

    malformed_success = ToolResult("first_tool", "false", "private malformed output SHOULD NOT APPEAR")
    malformed_verified, malformed_verification = verifier.verify(
        Plan("reject malformed success", [PlannedAction("first_tool")]),
        [malformed_success],
    )
    if malformed_verified or "malformed success flag" not in malformed_verification:
        raise SystemExit(f"Verifier accepted a malformed success flag: {malformed_verification}")
    if "SHOULD NOT APPEAR" in malformed_verification or "private malformed output" in malformed_verification:
        raise SystemExit(f"Verifier leaked malformed receipt output: {malformed_verification}")

    binding_failures = [
        (complete_results[:1], "count mismatch"),
        (complete_results + [ToolResult("extra_tool", True, "extra")], "count mismatch"),
        (list(reversed(complete_results)), "did not match planned tool"),
        ([ToolResult("wrong_tool", True, "wrong"), complete_results[1]], "did not match planned tool"),
    ]
    for results, expected_detail in binding_failures:
        verified, verification = verifier.verify(successful_plan, results)
        if verified or expected_detail not in verification:
            raise SystemExit(f"Verifier accepted an unbound result set: {results!r} -> {verification}")

    runtime = object.__new__(JarvisRuntime)
    noisy_output = "/\x55sers/example/Desktop/Claude code/AI agents/jarvis-v2/secret.txt\n/var/folders/zc/runtime-secret.log /tmp/runtime-secret.log " + ("line detail " * 80)
    trace = runtime._result_trace(ToolResult("noisy_tool", False, noisy_output, {"failure_kind": "tool_error"}))
    preview = trace.get("output_preview")
    if not isinstance(preview, str) or len(preview) > MAX_RUNTIME_OUTPUT_PREVIEW_CHARS:
        raise SystemExit(f"Runtime output preview should be bounded: {trace}")
    if any(fragment in preview for fragment in ["/\x55sers/", "Desktop/Claude code", "/var/folders/", "/tmp/"]) or "\n" in preview:
        raise SystemExit(f"Runtime output preview should scrub local paths and normalize whitespace: {trace}")
    if trace.get("failure_kind") != "tool_error" or trace.get("tool_name") != "noisy_tool":
        raise SystemExit(f"Runtime result trace should preserve ordinary receipt metadata: {trace}")

    action_trace = runtime._action_trace(
        PlannedAction(
            "explode",
            {
                "path": "/\x55sers/example/Desktop/Claude code/AI agents/jarvis-v2/secret.txt",
                "notes": "line one\n/var/folders/zc/action-secret.log /tmp/action-secret.log " + ("long detail " * 60),
            },
            "exercise action trace args",
        )
    )
    if action_trace.get("arg_keys") != ["notes", "path"]:
        raise SystemExit(f"Runtime action trace should expose stable arg keys: {action_trace}")
    action_args = action_trace.get("args")
    if not isinstance(action_args, dict) or set(action_args) != {"notes", "path"}:
        raise SystemExit(f"Runtime action trace should keep a preview dict shape: {action_trace}")
    joined_args = " ".join(str(value) for value in action_args.values())
    if any(fragment in joined_args for fragment in ["/\x55sers/", "Desktop/Claude code", "/var/folders/", "/tmp/"]) or "\n" in joined_args:
        raise SystemExit(f"Runtime action trace args should scrub paths and normalize whitespace: {action_trace}")
    if any(len(str(value)) > MAX_RUNTIME_OUTPUT_PREVIEW_CHARS for value in action_args.values()):
        raise SystemExit(f"Runtime action trace args should bound long values: {action_trace}")

    audit_metadata = runtime._tool_run_audit_metadata(
        ToolResult(
            "noisy_tool",
            False,
            "held",
            {
                "failure_kind": "approval_required",
                "planned_args": {"path": "/\x55sers/example/Desktop/Claude code/AI agents/jarvis-v2/secret.txt"},
                "planner_reason": "review /\x55sers/example/Desktop/Claude code/AI agents/jarvis-v2/secret.txt\n/var/folders/zc/planner-secret.log /tmp/planner-secret.log " + ("reason detail " * 80),
            },
        )
    )
    planner_reason = audit_metadata.get("planner_reason")
    if not isinstance(planner_reason, str) or len(planner_reason) > 260:
        raise SystemExit(f"Runtime audit planner reason should be bounded: {audit_metadata}")
    if any(fragment in planner_reason for fragment in ["/\x55sers/", "Desktop/Claude code", "/var/folders/", "/tmp/"]) or "\n" in planner_reason:
        raise SystemExit(f"Runtime audit planner reason should scrub paths and normalize whitespace: {audit_metadata}")
    if audit_metadata.get("planned_arg_keys") != ["path"]:
        raise SystemExit(f"Runtime audit metadata should preserve planned arg keys: {audit_metadata}")


if __name__ == "__main__":
    main()
