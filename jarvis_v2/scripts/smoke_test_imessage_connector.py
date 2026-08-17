"""Smoke tests for the iMessage send/read connector (mocked, no real Messages DB)."""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace

from jarvis_v2.agent.types import RiskLevel
from jarvis_v2.config import load_config
from jarvis_v2.tools import imessage_connector as ic


LOCAL_PATH_FRAGMENTS = ("/\x55sers/", "/private/", "/var/folders/", "/tmp/")
NO_AUTHORITY_FLAGS = {
    "authorizes_execution": False,
    "authorizes_completion_claim": False,
    "approval_granted": False,
}


def _tools():
    return {tool.name: tool for tool in ic.make_imessage_tools(load_config())}


def assert_exact_metadata_bool_helper() -> None:
    cases = [
        (True, False, True),
        (False, True, False),
        ("true", False, False),
        ("false", False, False),
        (1, False, False),
        (None, True, True),
        (None, False, False),
    ]
    for value, default, expected in cases:
        observed = ic._metadata_bool(value, default=default)
        if observed is not expected:
            raise SystemExit(f"iMessage _metadata_bool accepted non-exact value {value!r}: expected {expected}, got {observed}")


def _leaks_local_path(value) -> bool:
    return any(fragment in str(value) for fragment in LOCAL_PATH_FRAGMENTS)


def assert_outcome_unknown_recovery(result, label: str) -> None:
    expected_output = ic._post_attempt_send_recovery_guidance()
    if result.ok or result.output != expected_output:
        raise SystemExit(f"{label} missed fixed outcome-unknown output: {result}")
    expected_declaration = {
        "version": 1,
        "action": expected_output,
        "commands": ["setup check"],
    }
    if result.metadata.get("recovery_guidance") != expected_declaration:
        raise SystemExit(f"{label} recovery declaration drifted: {result.metadata}")
    expected = {
        "outcome_known": False,
        "outcome_unknown": True,
        "execution_outcome_unknown": True,
        "side_effect_possible": True,
        "retry_safe": False,
        "automatic_retry_allowed": False,
        "authorizes_retry": False,
        "next_command": "setup check",
        "recovery_commands": ["setup check"],
    }
    for key, value in expected.items():
        if result.metadata.get(key) != value:
            raise SystemExit(f"{label} outcome/retry field {key} drifted: {result.metadata}")


def assert_known_not_sent_recovery(result, label: str) -> None:
    action = ic.KNOWN_NOT_SENT_SEND_RECOVERY_ACTION
    if action not in result.output:
        raise SystemExit(f"{label} hid the canonical recovery action: {result}")
    if result.metadata.get("recovery_guidance") != {
        "version": 1,
        "action": action,
        "commands": ["setup check"],
    }:
        raise SystemExit(f"{label} recovery declaration drifted: {result.metadata}")
    expected = {
        "outcome_known": True,
        "outcome_unknown": False,
        "execution_outcome_unknown": False,
        "side_effect_possible": False,
        "automatic_retry_allowed": False,
        "authorizes_retry": False,
        "next_command": "setup check",
        "recovery_commands": ["setup check"],
    }
    for key, value in expected.items():
        if result.metadata.get(key) != value:
            raise SystemExit(f"{label} recovery field {key} drifted: {result.metadata}")


def assert_personal_read_recovery(result, label: str) -> None:
    action = ic.PERSONAL_READ_RECOVERY_ACTION
    if result.ok or action not in result.output:
        raise SystemExit(f"{label} hid the canonical read recovery action: {result}")
    if result.metadata.get("recovery_guidance") != {
        "version": 1,
        "action": action,
        "commands": ["setup check"],
    }:
        raise SystemExit(f"{label} read recovery declaration drifted: {result.metadata}")
    expected = {
        "outcome_known": True,
        "outcome_unknown": False,
        "execution_outcome_unknown": False,
        "side_effect_possible": False,
        "retry_safe": True,
        "automatic_retry_allowed": False,
        "authorizes_retry": False,
        "next_command": "setup check",
        "recovery_commands": ["setup check"],
    }
    for key, value in expected.items():
        if result.metadata.get(key) != value:
            raise SystemExit(f"{label} read recovery field {key} drifted: {result.metadata}")


def assert_imessage_send_handoff(metadata: dict, label: str, *, status: str) -> dict:
    if metadata.get("imessage_send_handoff_ready") is not True:
        raise SystemExit(f"{label} missed flat iMessage handoff readiness: {metadata}")
    handoff = metadata.get("imessage_send_handoff")
    if not isinstance(handoff, dict):
        raise SystemExit(f"{label} missed iMessage send handoff: {metadata}")
    if handoff.get("source") != "send_imessage":
        raise SystemExit(f"{label} iMessage handoff source wrong: {handoff}")
    if handoff.get("handoff_ready") is not True:
        raise SystemExit(f"{label} iMessage nested handoff readiness missing: {handoff}")
    if handoff.get("status") != status:
        raise SystemExit(f"{label} iMessage handoff status wrong: {handoff}")
    send_attempted = bool(handoff.get("send_attempted"))
    expected_changed = ["imessage_send_attempt"] if send_attempted else []
    for key, expected_value in [
        ("ready_for_operator", True),
        ("state_changed", send_attempted),
        ("content_in_handoff", False),
        ("authorizes_execution", False),
        ("authorizes_completion_claim", False),
        ("approval_granted", False),
    ]:
        if metadata.get(key) is not expected_value:
            raise SystemExit(f"{label} flat {key} should be {expected_value}: {metadata}")
        if handoff.get(key) is not expected_value:
            raise SystemExit(f"{label} handoff {key} should be {expected_value}: {handoff}")
    if metadata.get("imessage_send_ready_for_operator") != handoff.get("ready_for_operator"):
        raise SystemExit(f"{label} iMessage ready alias parity failed: {metadata} vs {handoff}")
    if metadata.get("imessage_send_state_changed") != handoff.get("state_changed"):
        raise SystemExit(f"{label} iMessage state alias parity failed: {metadata} vs {handoff}")
    if metadata.get("imessage_send_content_in_handoff") != handoff.get("content_in_handoff"):
        raise SystemExit(f"{label} iMessage content alias parity failed: {metadata} vs {handoff}")
    for key in NO_AUTHORITY_FLAGS:
        alias = f"imessage_send_{key}"
        if metadata.get(alias) is not False or metadata.get(alias) != handoff.get(key):
            raise SystemExit(f"{label} iMessage no-authority alias parity failed for {alias}: {metadata} vs {handoff}")
    if metadata.get("changed") != expected_changed or handoff.get("changed") != expected_changed:
        raise SystemExit(f"{label} changed should reflect attempted send only: {handoff} vs {metadata}")
    if metadata.get("imessage_send_changed") != expected_changed:
        raise SystemExit(f"{label} iMessage changed alias parity failed: {metadata} vs {handoff}")
    if handoff.get("to") != metadata.get("to", handoff.get("to")):
        raise SystemExit(f"{label} iMessage handoff recipient parity failed: {metadata}")
    if handoff.get("message_chars") != metadata.get("message_chars"):
        raise SystemExit(f"{label} iMessage handoff message length parity failed: {metadata}")
    if handoff.get("content_in_metadata") is not False:
        raise SystemExit(f"{label} iMessage handoff should not carry message content: {handoff}")
    if metadata.get("imessage_send_content_in_metadata") != handoff.get("content_in_metadata"):
        raise SystemExit(f"{label} iMessage content metadata alias parity failed: {metadata} vs {handoff}")
    if handoff.get("approval_required_before_execution") is not True or handoff.get("manual_review_required") is not True:
        raise SystemExit(f"{label} iMessage handoff should preserve approval/manual-review gate: {handoff}")
    expected_commands = {
        "outcome_unknown": ["recent tool runs", "execution recovery"],
        "failed": ["setup check"],
        "sender_confirmed": ["recent tool runs"],
    }.get(status, ["safe next actions"])
    if handoff.get("next_safe_command") != expected_commands[0]:
        raise SystemExit(f"{label} iMessage next safe command wrong: {handoff}")
    if handoff.get("next_safe_commands") != expected_commands:
        raise SystemExit(f"{label} iMessage safe-command list wrong: {handoff}")
    if handoff.get("next_safe_command_count") != len(expected_commands):
        raise SystemExit(f"{label} iMessage safe-command count wrong: {handoff}")
    if metadata.get("imessage_send_next_safe_command") != handoff.get("next_safe_command"):
        raise SystemExit(f"{label} iMessage next-command alias parity failed: {metadata} vs {handoff}")
    if metadata.get("imessage_send_next_safe_commands") != expected_commands:
        raise SystemExit(f"{label} iMessage next-commands alias parity failed: {metadata} vs {handoff}")
    if metadata.get("imessage_send_next_safe_command_count") != len(expected_commands):
        raise SystemExit(f"{label} iMessage next-command count alias parity failed: {metadata} vs {handoff}")
    if _leaks_local_path(handoff):
        raise SystemExit(f"{label} iMessage handoff leaked local path: {handoff}")
    boundaries = handoff.get("boundaries")
    if not isinstance(boundaries, dict):
        raise SystemExit(f"{label} iMessage handoff missed boundaries: {handoff}")
    contact_lookup_attempted = ic._metadata_bool(handoff.get("contact_lookup_attempted"))
    expected = {
        "calls_model": False,
        "calls_external_service": False,
        "executes_tools": False,
        "reads_personal_data": contact_lookup_attempted,
        "reads_private_data": False,
        "executes_side_effect": send_attempted,
        "external_side_effect": send_attempted,
        "writes_files": False,
        "writes_database": False,
        "writes_memory": False,
        "writes_notes": False,
        "queues_approval": False,
        "requires_approval": send_attempted,
        "controls_computer": False,
        **NO_AUTHORITY_FLAGS,
    }
    if metadata.get("imessage_send_boundaries") != boundaries:
        raise SystemExit(f"{label} iMessage boundary alias parity failed: {metadata} vs {handoff}")
    for key, value in expected.items():
        if boundaries.get(key) is not value:
            raise SystemExit(f"{label} iMessage boundary {key} wrong: {handoff}")
        if metadata.get(key, value) is not value:
            raise SystemExit(f"{label} iMessage flat boundary {key} wrong: {metadata}")
    return handoff


def assert_imessage_recent_handoff(metadata: dict, label: str, *, status: str) -> dict:
    if metadata.get("imessage_recent_handoff_ready") is not True:
        raise SystemExit(f"{label} missed flat iMessage recent handoff readiness: {metadata}")
    handoff = metadata.get("imessage_recent_handoff")
    if not isinstance(handoff, dict):
        raise SystemExit(f"{label} missed iMessage recent handoff: {metadata}")
    if handoff.get("source") != "read_recent_imessages":
        raise SystemExit(f"{label} recent handoff source wrong: {handoff}")
    if handoff.get("handoff_ready") is not True:
        raise SystemExit(f"{label} recent nested handoff readiness missing: {handoff}")
    for key, expected_value in [
        ("ready_for_operator", True),
        ("state_changed", False),
        ("content_in_handoff", True),
        *NO_AUTHORITY_FLAGS.items(),
    ]:
        if metadata.get(key) is not expected_value:
            raise SystemExit(f"{label} flat {key} should be {expected_value}: {metadata}")
        if handoff.get(key) is not expected_value:
            raise SystemExit(f"{label} recent handoff {key} should be {expected_value}: {handoff}")
    if metadata.get("imessage_recent_ready_for_operator") != handoff.get("ready_for_operator"):
        raise SystemExit(f"{label} recent ready alias parity failed: {metadata} vs {handoff}")
    if metadata.get("imessage_recent_state_changed") != handoff.get("state_changed"):
        raise SystemExit(f"{label} recent state alias parity failed: {metadata} vs {handoff}")
    if metadata.get("imessage_recent_content_in_handoff") != handoff.get("content_in_handoff"):
        raise SystemExit(f"{label} recent content alias parity failed: {metadata} vs {handoff}")
    for key in NO_AUTHORITY_FLAGS:
        alias = f"imessage_recent_{key}"
        if metadata.get(alias) is not False or metadata.get(alias) != handoff.get(key):
            raise SystemExit(f"{label} recent no-authority alias parity failed for {alias}: {metadata} vs {handoff}")
    if metadata.get("changed") != [] or handoff.get("changed") != []:
        raise SystemExit(f"{label} recent changed should be empty: {handoff} vs {metadata}")
    if metadata.get("imessage_recent_changed") != []:
        raise SystemExit(f"{label} recent changed alias parity failed: {metadata} vs {handoff}")
    if handoff.get("status") != status:
        raise SystemExit(f"{label} recent handoff status wrong: {handoff}")
    if handoff.get("limit") != metadata.get("limit"):
        raise SystemExit(f"{label} recent handoff limit parity failed: {metadata}")
    if handoff.get("message_count") != metadata.get("count"):
        raise SystemExit(f"{label} recent handoff count parity failed: {metadata}")
    messages = handoff.get("messages")
    if not isinstance(messages, list) or len(messages) != handoff.get("message_count"):
        raise SystemExit(f"{label} recent handoff message rows wrong: {handoff}")
    if handoff.get("content_in_metadata") is not True or handoff.get("content_is_preview_only") is not True:
        raise SystemExit(f"{label} recent handoff should declare preview-only content: {handoff}")
    if metadata.get("imessage_recent_content_in_metadata") != handoff.get("content_in_metadata"):
        raise SystemExit(f"{label} recent content metadata alias parity failed: {metadata} vs {handoff}")
    expected_next = f"read recent imessages limit {handoff.get('limit')}"
    if handoff.get("next_safe_command") != expected_next:
        raise SystemExit(f"{label} recent next safe command wrong: {handoff}")
    expected_commands = [expected_next]
    if handoff.get("next_safe_commands") != expected_commands:
        raise SystemExit(f"{label} recent safe-command list wrong: {handoff}")
    if handoff.get("next_safe_command_count") != len(expected_commands):
        raise SystemExit(f"{label} recent safe-command count wrong: {handoff}")
    if metadata.get("imessage_recent_next_safe_command") != expected_next:
        raise SystemExit(f"{label} recent next-command alias parity failed: {metadata} vs {handoff}")
    if metadata.get("imessage_recent_next_safe_commands") != expected_commands:
        raise SystemExit(f"{label} recent next-commands alias parity failed: {metadata} vs {handoff}")
    if metadata.get("imessage_recent_next_safe_command_count") != len(expected_commands):
        raise SystemExit(f"{label} recent next-command count alias parity failed: {metadata} vs {handoff}")
    if _leaks_local_path(handoff):
        raise SystemExit(f"{label} recent handoff leaked local path: {handoff}")
    boundaries = handoff.get("boundaries")
    if not isinstance(boundaries, dict):
        raise SystemExit(f"{label} recent handoff missed boundaries: {handoff}")
    if boundaries.get("reads_personal_data") is not True:
        raise SystemExit(f"{label} recent handoff should declare personal-data read: {handoff}")
    if metadata.get("imessage_recent_boundaries") != boundaries:
        raise SystemExit(f"{label} recent boundary alias parity failed: {metadata} vs {handoff}")
    for key in [
        "calls_model",
        "calls_external_service",
        "executes_tools",
        "reads_private_data",
        "executes_side_effect",
        "external_side_effect",
        "writes_files",
        "writes_database",
        "writes_memory",
        "writes_notes",
        "queues_approval",
        "controls_computer",
        *NO_AUTHORITY_FLAGS.keys(),
    ]:
        if boundaries.get(key) or metadata.get(key):
            raise SystemExit(f"{label} recent read should not perform {key}: {metadata}")
    return handoff


class FakeConnection:
    def __init__(self, rows):
        self.rows = rows
        self.closed = False
        self.last_limit = None

    def execute(self, _query, params):
        self.last_limit = params[0]
        return self

    def fetchall(self):
        return self.rows[: self.last_limit]

    def close(self):
        self.closed = True


def _with_fake_sqlite(rows):
    conn = FakeConnection(rows)
    module = types.ModuleType("sqlite3")
    module.connect = lambda _path: conn
    return module, conn


def test_send_validation_and_risk() -> None:
    tools = _tools()
    if tools["send_imessage"].risk != RiskLevel.HIGH_RISK:
        raise SystemExit("send_imessage must stay HIGH_RISK")
    missing_to = tools["send_imessage"].handler({"message": "hello"})
    if missing_to.ok or missing_to.metadata.get("executes_side_effect") or missing_to.metadata.get("requires_approval"):
        raise SystemExit(f"send_imessage missing recipient should be inert validation metadata: {missing_to.metadata}")
    missing_to_handoff = assert_imessage_send_handoff(missing_to.metadata, "iMessage missing-recipient", status="refused")
    if missing_to_handoff.get("reason") != "missing_recipient" or missing_to_handoff.get("send_attempted"):
        raise SystemExit(f"send_imessage missing recipient handoff should be inert refusal: {missing_to_handoff}")
    missing_body = tools["send_imessage"].handler({"to": "+15551234567"})
    if missing_body.ok or missing_body.metadata.get("external_side_effect") or missing_body.metadata.get("requires_approval"):
        raise SystemExit(f"send_imessage missing body should be inert validation metadata: {missing_body.metadata}")
    missing_body_handoff = assert_imessage_send_handoff(missing_body.metadata, "iMessage missing-body", status="refused")
    if missing_body_handoff.get("reason") != "missing_message" or missing_body_handoff.get("send_attempted"):
        raise SystemExit(f"send_imessage missing body handoff should be inert refusal: {missing_body_handoff}")


def test_send_redacts_path_shaped_recipient_metadata() -> None:
    tools = _tools()
    missing_body = tools["send_imessage"].handler({"to": "/\x55sers/example/Desktop/Claude code/imessage-target"})
    if missing_body.ok or missing_body.metadata.get("to") != "<local-path>":
        raise SystemExit(f"send_imessage should redact path-shaped missing-body recipients: {missing_body.metadata}")
    handoff = assert_imessage_send_handoff(missing_body.metadata, "iMessage path-shaped missing-body", status="refused")
    if handoff.get("to") != "<local-path>" or handoff.get("reason") != "missing_message":
        raise SystemExit(f"send_imessage path-shaped handoff should redact recipient: {handoff}")
    if _leaks_local_path(missing_body.output) or _leaks_local_path(missing_body.metadata):
        raise SystemExit(f"send_imessage validation leaked path-shaped recipient: {missing_body.output} {missing_body.metadata}")


def test_send_escapes_applescript_strings_without_live_send() -> None:
    calls = []

    def fake_run(cmd, capture_output, text, timeout):
        calls.append((cmd, capture_output, text, timeout))
        return SimpleNamespace(returncode=0, stdout="sent", stderr="")

    old_run = ic.subprocess.run
    try:
        ic.subprocess.run = fake_run
        out = _tools()["send_imessage"].handler({"to": 'alex"mac\\home@example.com', "message": 'say "hi" \\ ok'})
    finally:
        ic.subprocess.run = old_run

    if not out.ok:
        raise SystemExit(f"mocked send should succeed: {out.output}")
    handoff = assert_imessage_send_handoff(out.metadata, "iMessage mocked send", status="sender_confirmed")
    if handoff.get("send_attempted") is not True or handoff.get("reason"):
        raise SystemExit(f"send_imessage mocked send handoff should preserve successful attempt: {handoff}")
    if out.metadata.get("requires_approval") is not True or handoff.get("boundaries", {}).get("requires_approval") is not True:
        raise SystemExit(f"send_imessage mocked send should preserve approval requirement parity: {out.metadata}")
    if (
        out.metadata.get("sender_side_send_confirmed") is not True
        or out.metadata.get("recipient_delivery_confirmed") is not False
        or out.metadata.get("delivery_confirmed") is not False
        or out.metadata.get("delivery_evidence") != "sender_side_only"
        or out.metadata.get("recipient_confirmation_required") is not True
        or "recipient delivery is not confirmed" not in out.output
        or "iMessage sent to" in out.output
    ):
        raise SystemExit(f"iMessage sender receipt overclaimed recipient delivery: {out}")
    if not calls:
        raise SystemExit("send_imessage did not invoke osascript")
    cmd, capture_output, text_mode, timeout = calls[0]
    if cmd[:2] != ["osascript", "-e"] or capture_output is not True or text_mode is not True or timeout != 15:
        raise SystemExit(f"send_imessage used unexpected subprocess args: {calls[0]}")
    script = cmd[2]
    if 'buddy "alex\\"mac\\\\home@example.com"' not in script:
        raise SystemExit(f"recipient was not AppleScript-escaped safely: {script}")
    if 'send "say \\"hi\\" \\\\ ok" to targetBuddy' not in script:
        raise SystemExit(f"message was not AppleScript-escaped safely: {script}")


def test_send_failure_is_clean_without_live_send() -> None:
    def fake_run(cmd, capture_output, text, timeout):
        return SimpleNamespace(returncode=1, stdout="", stderr="raw osascript failure")

    old_run = ic.subprocess.run
    try:
        ic.subprocess.run = fake_run
        out = _tools()["send_imessage"].handler({"to": "+15551234567", "message": "hello"})
    finally:
        ic.subprocess.run = old_run

    assert_outcome_unknown_recovery(out, "generic iMessage send failure")
    if "raw osascript failure" in out.output or "iMessage error" in out.output:
        raise SystemExit(f"send_imessage failure should not leak raw exceptions: {out.output}")
    if out.metadata.get("to") != "+15551234567" or out.metadata.get("exception_type") != "IMessageSendError":
        raise SystemExit(f"send_imessage failure should preserve bounded diagnostic metadata: {out.metadata}")
    if out.metadata.get("imessage_send_stage") != "osascript_failed":
        raise SystemExit(f"send_imessage failure should expose a channel-health stage alias: {out.metadata}")
    handoff = assert_imessage_send_handoff(out.metadata, "iMessage send failure", status="outcome_unknown")
    if handoff.get("reason") != "send_outcome_unknown" or handoff.get("send_attempted") is not True:
        raise SystemExit(f"send_imessage failure handoff should preserve outcome uncertainty: {handoff}")
    if out.metadata.get("requires_approval") is not True or handoff.get("boundaries", {}).get("requires_approval") is not True:
        raise SystemExit(f"send_imessage failure should preserve approval requirement parity: {out.metadata}")


def test_send_failure_stages_are_actionable_and_redacted() -> None:
    cases = (
        (
            "execution error: Not authorized to send Apple events to Messages. (-1743)",
            "macos_automation_permission",
            "Privacy & Security > Automation",
        ),
        (
            "Messages got an error: Can't get buddy \"+15551234567\" of service 1. (-1728)",
            "imessage_recipient_unavailable",
            "registered for iMessage",
        ),
    )
    for raw_error, expected_stage, expected_guidance in cases:
        stage, detail = ic._imessage_error_stage(raw_error)
        if stage != expected_stage or expected_guidance not in detail:
            raise SystemExit(f"iMessage error classifier drifted: {stage!r} {detail!r}")
        failure_stage, output = ic._imessage_send_failure(ic.IMessageSendError(stage, detail))
        if failure_stage != expected_stage or expected_stage not in output or expected_guidance not in output:
            raise SystemExit(f"iMessage staged recovery guidance drifted: {failure_stage!r} {output!r}")
        if raw_error in output or "(-1743)" in output or "+15551234567" in output:
            raise SystemExit(f"iMessage staged recovery guidance leaked raw diagnostics: {output!r}")

    timeout_stage, timeout_output = ic._imessage_send_failure(ic.subprocess.TimeoutExpired(["osascript"], 15))
    if timeout_stage != "automation_timeout" or "timed out" not in timeout_output:
        raise SystemExit(f"iMessage timeout classification drifted: {timeout_stage!r} {timeout_output!r}")

    original_send = ic._send_imessage
    try:
        ic._send_imessage = lambda _to, _message: (_ for _ in ()).throw(
            ic.IMessageSendError(
                "macos_automation_permission",
                "In System Settings > Privacy & Security > Automation, allow the Jarvis Python process "
                "to control Messages, then retry.",
            )
        )
        out = _tools()["send_imessage"].handler({"to": "+15551234567", "message": "hello"})
    finally:
        ic._send_imessage = original_send
    if out.ok or out.metadata.get("failure_stage") != "macos_automation_permission":
        raise SystemExit(f"iMessage tool receipt missed staged failure metadata: {out.output} {out.metadata}")
    assert_known_not_sent_recovery(out, "iMessage permission failure")
    if (
        out.metadata.get("outcome_known") is not True
        or out.metadata.get("outcome_unknown") is not False
        or out.metadata.get("side_effect_possible") is not False
        or out.metadata.get("automatic_retry_allowed") is not False
        or out.metadata.get("authorizes_retry") is not False
    ):
        raise SystemExit(f"iMessage proven pre-send failure lost known-no-send truth: {out.metadata}")
    handoff = assert_imessage_send_handoff(out.metadata, "iMessage permission failure", status="failed")
    if handoff.get("reason") != "send_error":
        raise SystemExit(f"iMessage permission failure handoff drifted: {handoff}")
    if "Privacy & Security > Automation" not in out.output or "+15551234567" in out.output:
        raise SystemExit(f"iMessage tool staged guidance leaked recipient data: {out.output!r}")


def test_send_resolves_contact_names_before_live_send() -> None:
    calls = []
    original_send = ic._send_imessage
    original_resolve = ic.contacts_connector.resolve_contact
    try:
        ic._send_imessage = lambda to, message: calls.append((to, message)) or ""
        ic.contacts_connector.resolve_contact = lambda query: [
            ic.contacts_connector.ContactMatch("Fixture Example", phone="+14155550100")
        ]
        out = _tools()["send_imessage"].handler({"to": "fixture", "message": "hello"})
    finally:
        ic._send_imessage = original_send
        ic.contacts_connector.resolve_contact = original_resolve
    if not out.ok or calls != [("+14155550100", "hello")]:
        raise SystemExit(f"send_imessage should resolve one contact before sending: {out} {calls}")
    if out.metadata.get("to") != "fixture" or out.metadata.get("resolved_to") != "+14155550100":
        raise SystemExit(f"send_imessage should preserve original and resolved recipients: {out.metadata}")
    handoff = assert_imessage_send_handoff(out.metadata, "iMessage resolved contact", status="sender_confirmed")
    if handoff.get("contact_resolution_status") != "resolved" or not handoff.get("contact_lookup_attempted"):
        raise SystemExit(f"send_imessage resolved contact handoff wrong: {handoff}")


def test_send_ambiguous_contact_refuses_without_live_send() -> None:
    calls = []
    original_send = ic._send_imessage
    original_resolve = ic.contacts_connector.resolve_contact
    try:
        ic._send_imessage = lambda to, message: calls.append((to, message)) or ""
        ic.contacts_connector.resolve_contact = lambda query: [
            ic.contacts_connector.ContactMatch("Fixture Example", phone="+14155550100"),
            ic.contacts_connector.ContactMatch("Fixture Kim", phone="+14155550101"),
        ]
        out = _tools()["send_imessage"].handler({"to": "fixture", "message": "hello"})
    finally:
        ic._send_imessage = original_send
        ic.contacts_connector.resolve_contact = original_resolve
    if out.ok or calls:
        raise SystemExit(f"send_imessage ambiguous contact must not send: {out} {calls}")
    if "Fixture Example" not in out.output or "Fixture Kim" not in out.output:
        raise SystemExit(f"send_imessage ambiguous contact should ask which contact: {out.output}")
    handoff = assert_imessage_send_handoff(out.metadata, "iMessage ambiguous contact", status="refused")
    if handoff.get("contact_resolution_status") != "ambiguous" or handoff.get("send_attempted"):
        raise SystemExit(f"send_imessage ambiguous handoff wrong: {handoff}")


def test_send_missing_contact_refuses_without_live_send() -> None:
    calls = []
    original_send = ic._send_imessage
    original_resolve = ic.contacts_connector.resolve_contact
    try:
        ic._send_imessage = lambda to, message: calls.append((to, message)) or ""
        ic.contacts_connector.resolve_contact = lambda query: []
        out = _tools()["send_imessage"].handler({"to": "fixture", "message": "hello"})
    finally:
        ic._send_imessage = original_send
        ic.contacts_connector.resolve_contact = original_resolve
    if out.ok or calls or "couldn't find" not in out.output.lower():
        raise SystemExit(f"send_imessage missing contact must refuse without sending: {out} {calls}")
    handoff = assert_imessage_send_handoff(out.metadata, "iMessage missing contact", status="refused")
    if handoff.get("contact_resolution_status") != "not_found" or handoff.get("send_attempted"):
        raise SystemExit(f"send_imessage missing contact handoff wrong: {handoff}")


def test_send_contact_lookup_failure_refuses_without_live_send() -> None:
    calls = []
    original_send = ic._send_imessage
    original_resolve = ic.contacts_connector.resolve_contact
    try:
        ic._send_imessage = lambda to, message: calls.append((to, message)) or ""
        ic.contacts_connector.resolve_contact = lambda _query: (_ for _ in ()).throw(
            RuntimeError("Contacts unavailable")
        )
        out = _tools()["send_imessage"].handler({"to": "fixture", "message": "hello"})
    finally:
        ic._send_imessage = original_send
        ic.contacts_connector.resolve_contact = original_resolve
    if out.ok or calls or "couldn't access Contacts" not in out.output:
        raise SystemExit(f"send_imessage Contacts failure must refuse safely: {out} {calls}")
    handoff = assert_imessage_send_handoff(out.metadata, "iMessage unavailable contact lookup", status="refused")
    if (
        handoff.get("contact_resolution_status") != "unavailable"
        or handoff.get("contact_lookup_attempted") is not True
        or handoff.get("send_attempted")
    ):
        raise SystemExit(f"send_imessage unavailable-contact handoff wrong: {handoff}")


def test_send_explicit_handle_skips_contact_resolution() -> None:
    calls = []
    original_send = ic._send_imessage
    original_resolve = ic.contacts_connector.resolve_contact
    try:
        ic._send_imessage = lambda to, message: calls.append((to, message)) or ""

        def fail_resolve(_query):
            raise AssertionError("explicit handle should not resolve contacts")

        ic.contacts_connector.resolve_contact = fail_resolve
        out = _tools()["send_imessage"].handler({"to": "+15551234567", "message": "hello"})
    finally:
        ic._send_imessage = original_send
        ic.contacts_connector.resolve_contact = original_resolve
    if not out.ok or calls != [("+15551234567", "hello")]:
        raise SystemExit(f"send_imessage explicit handle should send as-is: {out} {calls}")
    handoff = assert_imessage_send_handoff(out.metadata, "iMessage explicit handle", status="sender_confirmed")
    if handoff.get("contact_resolution_status") != "skipped" or handoff.get("contact_lookup_attempted"):
        raise SystemExit(f"send_imessage explicit handle handoff wrong: {handoff}")


def test_send_malformed_contact_lookup_metadata_defaults_closed() -> None:
    original_send = ic._send_imessage
    original_resolve_recipient = ic._resolve_send_recipient
    calls = []
    try:
        ic._send_imessage = lambda to, message: calls.append((to, message)) or ""
        ic._resolve_send_recipient = lambda _recipient: (
            None,
            {
                "original_to": "fixture",
                "contact_lookup_attempted": "false",
                "contact_resolution_status": "not_found",
                "contact_match_count": 0,
                "contact_candidates": [],
            },
            "not found",
        )
        refused = _tools()["send_imessage"].handler({"to": "fixture", "message": "hello"})
        ic._resolve_send_recipient = lambda _recipient: (
            "+14155550100",
            {
                "original_to": "fixture",
                "contact_lookup_attempted": "false",
                "contact_resolution_status": "resolved",
                "contact_match_count": 1,
                "contact_candidates": ["Fixture Example"],
            },
            "",
        )
        sent = _tools()["send_imessage"].handler({"to": "fixture", "message": "hello"})
    finally:
        ic._send_imessage = original_send
        ic._resolve_send_recipient = original_resolve_recipient

    if refused.ok or calls != [("+14155550100", "hello")]:
        raise SystemExit(f"malformed lookup metadata should refuse first and mock-send second only: {refused} {sent} {calls}")
    refused_handoff = assert_imessage_send_handoff(refused.metadata, "iMessage malformed lookup refusal", status="refused")
    if (
        refused.metadata.get("reads_personal_data") is not False
        or refused.metadata.get("contact_lookup_attempted") is not False
        or refused_handoff.get("contact_lookup_attempted") is not False
        or refused_handoff.get("boundaries", {}).get("reads_personal_data") is not False
    ):
        raise SystemExit(f"malformed refused lookup metadata should default closed: {refused.metadata}")

    if not sent.ok:
        raise SystemExit(f"malformed lookup success path should still allow mocked send to resolved handle: {sent}")
    sent_handoff = assert_imessage_send_handoff(sent.metadata, "iMessage malformed lookup sent", status="sender_confirmed")
    if (
        sent.metadata.get("reads_personal_data") is not False
        or sent.metadata.get("contact_lookup_attempted") is not False
        or sent_handoff.get("contact_lookup_attempted") is not False
        or sent_handoff.get("boundaries", {}).get("reads_personal_data") is not False
    ):
        raise SystemExit(f"malformed sent lookup metadata should default closed: {sent.metadata}")


def test_read_recent_imessages_mocked_limit_metadata() -> None:
    rows = [
        ("older private text", 0, "+15550000001", 1),
        (
            "newer private text near /\x55sers/example/Desktop/Claude code/imessage.txt "
            "plus /var/folders/zc/jarvis-proof.txt and /tmp/jarvis-proof.txt",
            1,
            "+15550000002",
            2,
        ),
    ]
    fake_sqlite, conn = _with_fake_sqlite(rows)
    old_sqlite = sys.modules.get("sqlite3")
    try:
        sys.modules["sqlite3"] = fake_sqlite
        out = _tools()["read_recent_imessages"].handler({"limit": "bad"})
    finally:
        if old_sqlite is None:
            sys.modules.pop("sqlite3", None)
        else:
            sys.modules["sqlite3"] = old_sqlite
    if not out.ok or "Recent 2 iMessages" not in out.output or "private text" not in out.output:
        raise SystemExit(f"read_recent_imessages mocked output wrong: {out.output}")
    if "/\x55sers/" in out.output or "/private/" in out.output or "/var/folders" in out.output or "/tmp/" in out.output or "<local-path>" not in out.output:
        raise SystemExit(f"read_recent_imessages should redact path-shaped message previews: {out.output}")
    if conn.last_limit != 10:
        raise SystemExit(f"read_recent_imessages should query with sanitized limit: {conn.last_limit}")
    if out.metadata.get("limit") != 10 or out.metadata.get("raw_limit") != "bad":
        raise SystemExit(f"read_recent_imessages should preserve sanitized raw limit metadata: {out.metadata}")
    if out.metadata.get("reads_personal_data") is not True or out.metadata.get("writes_files"):
        raise SystemExit(f"read_recent_imessages missed privacy/read-only metadata: {out.metadata}")
    handoff = assert_imessage_recent_handoff(out.metadata, "read_recent_imessages populated", status="ok")
    if handoff["messages"][0].get("direction") != "sent" or handoff["messages"][1].get("direction") != "received":
        raise SystemExit(f"read_recent_imessages handoff should preserve display order/direction: {handoff}")
    if "<local-path>" not in handoff["messages"][0].get("preview", ""):
        raise SystemExit(f"read_recent_imessages handoff should preserve redacted preview marker: {handoff}")

    fake_sqlite, conn = _with_fake_sqlite(rows)
    old_sqlite = sys.modules.get("sqlite3")
    try:
        sys.modules["sqlite3"] = fake_sqlite
        bool_out = _tools()["read_recent_imessages"].handler({"limit": False})
    finally:
        if old_sqlite is None:
            sys.modules.pop("sqlite3", None)
        else:
            sys.modules["sqlite3"] = old_sqlite
    if not bool_out.ok or conn.last_limit != 10:
        raise SystemExit(f"read_recent_imessages should query with default limit for boolean input: {conn.last_limit}")
    if bool_out.metadata.get("limit") != 10 or bool_out.metadata.get("raw_limit") != "False":
        raise SystemExit(f"read_recent_imessages should treat boolean limits as malformed defaults: {bool_out.metadata}")
    if bool_out.metadata.get("reads_personal_data") is not True or bool_out.metadata.get("writes_files"):
        raise SystemExit(f"read_recent_imessages boolean limit missed privacy/read-only metadata: {bool_out.metadata}")
    bool_handoff = assert_imessage_recent_handoff(bool_out.metadata, "read_recent_imessages boolean limit", status="ok")
    if bool_handoff.get("limit") != 10:
        raise SystemExit(f"read_recent_imessages boolean limit handoff should preserve sanitized limit: {bool_handoff}")

    fake_sqlite, conn = _with_fake_sqlite(rows)
    old_sqlite = sys.modules.get("sqlite3")
    try:
        sys.modules["sqlite3"] = fake_sqlite
        path_out = _tools()["read_recent_imessages"].handler({"limit": "/var/folders/zc/jarvis-imessage-limit"})
    finally:
        if old_sqlite is None:
            sys.modules.pop("sqlite3", None)
        else:
            sys.modules["sqlite3"] = old_sqlite
    if not path_out.ok or conn.last_limit != 10:
        raise SystemExit(f"read_recent_imessages should query with default limit for path-shaped input: {conn.last_limit}")
    if path_out.metadata.get("limit") != 10 or path_out.metadata.get("raw_limit") != "<local-path>":
        raise SystemExit(f"read_recent_imessages leaked local path in raw limit metadata: {path_out.metadata}")
    if path_out.metadata.get("reads_personal_data") is not True or path_out.metadata.get("writes_files"):
        raise SystemExit(f"read_recent_imessages path limit missed privacy/read-only metadata: {path_out.metadata}")
    path_handoff = assert_imessage_recent_handoff(path_out.metadata, "read_recent_imessages path limit", status="ok")
    if path_handoff.get("limit") != 10 or _leaks_local_path(path_handoff):
        raise SystemExit(f"read_recent_imessages path limit handoff should be redacted: {path_handoff}")


def test_read_recent_imessages_bounds_and_errors() -> None:
    fake_sqlite, conn = _with_fake_sqlite([])
    old_sqlite = sys.modules.get("sqlite3")
    try:
        sys.modules["sqlite3"] = fake_sqlite
        empty = _tools()["read_recent_imessages"].handler({"limit": 9999})
    finally:
        if old_sqlite is None:
            sys.modules.pop("sqlite3", None)
        else:
            sys.modules["sqlite3"] = old_sqlite
    if not empty.ok or empty.metadata.get("count") != 0 or empty.metadata.get("limit") != 50 or conn.last_limit != 50:
        raise SystemExit(f"read_recent_imessages should clamp empty reads: {empty.metadata}, limit={conn.last_limit}")
    empty_handoff = assert_imessage_recent_handoff(empty.metadata, "read_recent_imessages empty", status="empty")
    if empty_handoff.get("messages") != []:
        raise SystemExit(f"read_recent_imessages empty handoff should carry no rows: {empty_handoff}")

    module = types.ModuleType("sqlite3")
    module.connect = lambda _path: (_ for _ in ()).throw(RuntimeError("db unavailable"))
    old_sqlite = sys.modules.get("sqlite3")
    try:
        sys.modules["sqlite3"] = module
        failed = _tools()["read_recent_imessages"].handler({"limit": "l" * 120})
    finally:
        if old_sqlite is None:
            sys.modules.pop("sqlite3", None)
        else:
            sys.modules["sqlite3"] = old_sqlite
    if failed.ok or "iMessage could not read" not in failed.output:
        raise SystemExit(f"read_recent_imessages failure should be friendly: {failed.output}")
    assert_personal_read_recovery(failed, "read_recent_imessages failure")
    if "db unavailable" in failed.output or "iMessage read error" in failed.output:
        raise SystemExit(f"read_recent_imessages failure should not leak raw exceptions: {failed.output}")
    if failed.metadata.get("limit") != 10 or failed.metadata.get("raw_limit") != ("l" * 79 + "…"):
        raise SystemExit(f"read_recent_imessages should bound raw error metadata: {failed.metadata}")
    if failed.metadata.get("exception_type") != "RuntimeError":
        raise SystemExit(f"read_recent_imessages should preserve bounded diagnostic metadata: {failed.metadata}")
    failed_handoff = assert_imessage_recent_handoff(failed.metadata, "read_recent_imessages failure", status="unavailable")
    if failed_handoff.get("reason") != "read_error" or failed_handoff.get("messages") != []:
        raise SystemExit(f"read_recent_imessages failure handoff should preserve read error: {failed_handoff}")


def main() -> None:
    assert_exact_metadata_bool_helper()
    test_send_validation_and_risk()
    test_send_redacts_path_shaped_recipient_metadata()
    test_send_escapes_applescript_strings_without_live_send()
    test_send_failure_is_clean_without_live_send()
    test_send_failure_stages_are_actionable_and_redacted()
    test_send_resolves_contact_names_before_live_send()
    test_send_ambiguous_contact_refuses_without_live_send()
    test_send_missing_contact_refuses_without_live_send()
    test_send_contact_lookup_failure_refuses_without_live_send()
    test_send_explicit_handle_skips_contact_resolution()
    test_send_malformed_contact_lookup_metadata_defaults_closed()
    test_read_recent_imessages_mocked_limit_metadata()
    test_read_recent_imessages_bounds_and_errors()
    print("iMessage connector smoke passed")


if __name__ == "__main__":
    main()
