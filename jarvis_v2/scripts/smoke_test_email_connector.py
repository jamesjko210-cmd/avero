"""Smoke tests for the Gmail connector (validation + mocked SMTP, no real send)."""

from __future__ import annotations

import os
from unittest import mock

from jarvis_v2.agent.planner import RuleBasedPlanner
from jarvis_v2.agent.types import RiskLevel
from jarvis_v2.config import load_config
from jarvis_v2.tools import email_connector as ec


LOCAL_PATH_FRAGMENTS = ("/\x55sers/", "/private/", "/var/folders/", "/tmp/")
NO_AUTHORITY_FLAGS = {
    "authorizes_execution": False,
    "authorizes_completion_claim": False,
    "approval_granted": False,
}
NO_AUTHORITY_KEYS = tuple(NO_AUTHORITY_FLAGS)


def _tools():
    return {t.name: t for t in ec.make_email_tools(load_config())}


def _leaks_local_path(value) -> bool:
    return any(fragment in str(value) for fragment in LOCAL_PATH_FRAGMENTS)


def assert_known_not_sent_recovery(result, label: str) -> None:
    action = ec.KNOWN_NOT_SENT_SEND_RECOVERY_ACTION
    if result.ok or action not in result.output:
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
    action = ec.PERSONAL_READ_RECOVERY_ACTION
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


def assert_command_handoff_contract(
    metadata: dict,
    handoff: dict,
    label: str,
    *,
    prefix: str,
    next_safe_command: str,
    next_safe_commands: list[str] | None = None,
    state_changed: bool = False,
    changed: list[str] | None = None,
) -> None:
    expected_changed = changed or []
    if metadata.get(f"{prefix}_handoff_ready") is not True or handoff.get("handoff_ready") is not True:
        raise SystemExit(f"{label} missed handoff readiness aliases: {metadata} {handoff}")
    expected_command_list = next_safe_commands or [next_safe_command]
    expected_values = {
        "ready_for_operator": True,
        "state_changed": state_changed,
        "changed": expected_changed,
        "content_in_handoff": False,
        "next_safe_command": next_safe_command,
        "next_safe_commands": expected_command_list,
        "next_safe_command_count": len(expected_command_list),
        **NO_AUTHORITY_FLAGS,
    }
    for key, expected_value in expected_values.items():
        if handoff.get(key) != expected_value:
            raise SystemExit(f"{label} handoff {key} should be {expected_value}: {handoff}")
        if metadata.get(f"{prefix}_{key}") != expected_value:
            raise SystemExit(f"{label} flat {prefix}_{key} should be {expected_value}: {metadata}")
    for key in [
        "ready_for_operator",
        "state_changed",
        "changed",
        "content_in_handoff",
        *NO_AUTHORITY_KEYS,
    ]:
        if metadata.get(key) != expected_values[key]:
            raise SystemExit(f"{label} shared flat {key} should be {expected_values[key]}: {metadata}")
    boundaries = handoff.get("boundaries")
    if metadata.get(f"{prefix}_boundaries") != boundaries:
        raise SystemExit(f"{label} boundary alias should mirror nested boundaries: {metadata} {handoff}")


def assert_email_send_handoff(metadata: dict, label: str, *, status: str) -> dict:
    handoff = metadata.get("email_send_handoff")
    if not isinstance(handoff, dict):
        raise SystemExit(f"{label} missed email send handoff: {metadata}")
    if handoff.get("source") != "send_email":
        raise SystemExit(f"{label} email handoff source wrong: {handoff}")
    send_attempted = bool(handoff.get("send_attempted"))
    expected_commands = (
        ["recent tool runs", "execution recovery"]
        if status == "outcome_unknown"
        else ["setup check"]
        if status in {"failed", "unavailable"}
        else ["recent tool runs"]
        if status == "sent"
        else ["safe next actions"]
    )
    assert_command_handoff_contract(
        metadata,
        handoff,
        label,
        prefix="email_send",
        next_safe_command=expected_commands[0],
        next_safe_commands=expected_commands,
        state_changed=send_attempted,
        changed=["email_send_attempt"] if send_attempted else [],
    )
    if handoff.get("status") != status:
        raise SystemExit(f"{label} email handoff status wrong: {handoff}")
    if (
        handoff.get("next_safe_commands") != expected_commands
        or handoff.get("next_safe_command_count") != len(expected_commands)
    ):
        raise SystemExit(f"{label} email recovery command list drifted: {handoff}")
    if handoff.get("to") != metadata.get("to", handoff.get("to")):
        raise SystemExit(f"{label} email handoff recipient parity failed: {metadata}")
    if handoff.get("subject") != metadata.get("subject", handoff.get("subject")):
        raise SystemExit(f"{label} email handoff subject parity failed: {metadata}")
    if handoff.get("body_chars") != metadata.get("body_chars"):
        raise SystemExit(f"{label} email handoff body length parity failed: {metadata}")
    if handoff.get("content_in_metadata") is not False:
        raise SystemExit(f"{label} email handoff should not carry body content: {handoff}")
    if metadata.get("email_send_content_in_metadata") is not False:
        raise SystemExit(f"{label} email send content metadata alias should be false: {metadata}")
    if metadata.get("email_send_retry_safe") is not handoff.get("retry_safe"):
        raise SystemExit(f"{label} email retry-safe alias drifted: {metadata}")
    if metadata.get("email_send_outcome_known") is not handoff.get("outcome_known"):
        raise SystemExit(f"{label} email outcome-known alias drifted: {metadata}")
    if handoff.get("approval_required_before_execution") is not True or handoff.get("manual_review_required") is not True:
        raise SystemExit(f"{label} email handoff should preserve approval/manual-review gate: {handoff}")
    if _leaks_local_path(handoff):
        raise SystemExit(f"{label} email handoff leaked local path: {handoff}")
    boundaries = handoff.get("boundaries")
    if not isinstance(boundaries, dict):
        raise SystemExit(f"{label} email handoff missed boundaries: {handoff}")
    external_attempted = bool(handoff.get("external_attempted"))
    requires_approval = external_attempted or send_attempted
    if metadata.get("external_attempted", False) is not external_attempted:
        raise SystemExit(f"{label} email external-attempt parity failed: {metadata}")
    if metadata.get("send_attempted", False) is not send_attempted:
        raise SystemExit(f"{label} email send-attempt parity failed: {metadata}")
    if metadata.get("requires_approval", False) is not requires_approval:
        raise SystemExit(f"{label} email flat approval boundary should match execution attempt: {metadata}")
    if boundaries.get("requires_approval") is not requires_approval:
        raise SystemExit(f"{label} email handoff approval boundary should match execution attempt: {handoff}")
    if boundaries.get("calls_external_service") is not external_attempted:
        raise SystemExit(f"{label} email handoff external-service boundary should match SMTP attempt: {handoff}")
    if boundaries.get("executes_side_effect") is not send_attempted or boundaries.get("external_side_effect") is not send_attempted:
        raise SystemExit(f"{label} email handoff side-effect boundary should match SMTP send attempt: {handoff}")
    contact_lookup_attempted = ec._metadata_bool(handoff.get("contact_lookup_attempted"))
    for key in [
        "calls_model",
        "executes_tools",
        "reads_private_data",
        "writes_files",
        "writes_database",
        "writes_memory",
        "writes_notes",
        "queues_approval",
        "controls_computer",
        *NO_AUTHORITY_FLAGS.keys(),
    ]:
        if boundaries.get(key) or metadata.get(key):
            raise SystemExit(f"{label} email send should not perform {key}: {metadata}")
    if boundaries.get("reads_personal_data") is not contact_lookup_attempted:
        raise SystemExit(f"{label} email personal-data boundary should match contact lookup: {handoff}")
    if metadata.get("reads_personal_data", False) is not contact_lookup_attempted:
        raise SystemExit(f"{label} email flat personal-data boundary should match contact lookup: {metadata}")
    return handoff


def assert_email_read_handoff(metadata: dict, label: str, *, status: str) -> dict:
    handoff = metadata.get("email_read_handoff")
    if not isinstance(handoff, dict):
        raise SystemExit(f"{label} missed email read handoff: {metadata}")
    expected_next = "read emails" if handoff.get("retry_safe") else "read email body"
    assert_command_handoff_contract(
        metadata,
        handoff,
        label,
        prefix="email_read",
        next_safe_command=expected_next,
    )
    if handoff.get("source") != "read_emails" or handoff.get("status") != status:
        raise SystemExit(f"{label} email read handoff source/status wrong: {handoff}")
    for key in [
        "ready_for_operator",
        "state_changed",
        "changed",
        "content_in_handoff",
        "authorizes_execution",
        "authorizes_completion_claim",
        "approval_granted",
    ]:
        if handoff.get(key) != metadata.get(key):
            raise SystemExit(f"{label} email read handoff should mirror {key}: {handoff} vs {metadata}")
    if handoff.get("limit") != metadata.get("limit"):
        raise SystemExit(f"{label} email read handoff limit parity failed: {metadata}")
    if handoff.get("count") != metadata.get("count", 0):
        raise SystemExit(f"{label} email read handoff count parity failed: {metadata}")
    rows = handoff.get("rows")
    if not isinstance(rows, list) or handoff.get("row_count") != len(rows):
        raise SystemExit(f"{label} email read handoff row count wrong: {handoff}")
    if handoff.get("headers_only") is not True or handoff.get("body_in_metadata") is not False:
        raise SystemExit(f"{label} email read handoff should be headers-only and body-free: {handoff}")
    if metadata.get("email_read_body_in_metadata") is not False:
        raise SystemExit(f"{label} email read body metadata alias should be false: {metadata}")
    if _leaks_local_path(handoff):
        raise SystemExit(f"{label} email read handoff leaked local path: {handoff}")
    external_attempted = bool(handoff.get("external_attempted"))
    if metadata.get("external_attempted", False) is not external_attempted:
        raise SystemExit(f"{label} email read external-attempt parity failed: {metadata}")
    boundaries = handoff.get("boundaries")
    if not isinstance(boundaries, dict):
        raise SystemExit(f"{label} email read handoff missed boundaries: {handoff}")
    expected = {
        "calls_model": False,
        "calls_external_service": external_attempted,
        "executes_tools": False,
        "reads_personal_data": True,
        "reads_private_data": False,
        "executes_side_effect": False,
        "external_side_effect": False,
        "writes_files": False,
        "writes_database": False,
        "writes_memory": False,
        "writes_notes": False,
        "queues_approval": False,
        "requires_approval": False,
        "controls_computer": False,
        **NO_AUTHORITY_FLAGS,
    }
    for key, value in expected.items():
        if boundaries.get(key) is not value:
            raise SystemExit(f"{label} email read boundary {key} wrong: {handoff}")
        if metadata.get(key, value) is not value:
            raise SystemExit(f"{label} email read flat boundary {key} wrong: {metadata}")
    return handoff


def assert_email_search_handoff(metadata: dict, label: str, *, status: str) -> dict:
    handoff = metadata.get("email_search_handoff")
    if not isinstance(handoff, dict):
        raise SystemExit(f"{label} missed email search handoff: {metadata}")
    expected_next = "search emails" if handoff.get("retry_safe") else "read email body"
    assert_command_handoff_contract(
        metadata,
        handoff,
        label,
        prefix="email_search",
        next_safe_command=expected_next,
    )
    if handoff.get("source") != "search_emails" or handoff.get("status") != status:
        raise SystemExit(f"{label} email search handoff source/status wrong: {handoff}")
    for key in [
        "ready_for_operator",
        "state_changed",
        "changed",
        "content_in_handoff",
        "authorizes_execution",
        "authorizes_completion_claim",
        "approval_granted",
    ]:
        if handoff.get(key) != metadata.get(key):
            raise SystemExit(f"{label} email search handoff should mirror {key}: {handoff} vs {metadata}")
    if handoff.get("limit") != metadata.get("limit"):
        raise SystemExit(f"{label} email search handoff limit parity failed: {metadata}")
    if handoff.get("criterion") != metadata.get("criterion", handoff.get("criterion")):
        raise SystemExit(f"{label} email search criterion parity failed: {metadata}")
    if handoff.get("count") != metadata.get("count", 0):
        raise SystemExit(f"{label} email search count parity failed: {metadata}")
    rows = handoff.get("rows")
    if not isinstance(rows, list) or handoff.get("row_count") != len(rows):
        raise SystemExit(f"{label} email search row count wrong: {handoff}")
    if handoff.get("headers_only") is not True or handoff.get("body_in_metadata") is not False:
        raise SystemExit(f"{label} email search handoff should be headers-only and body-free: {handoff}")
    if metadata.get("email_search_body_in_metadata") is not False:
        raise SystemExit(f"{label} email search body metadata alias should be false: {metadata}")
    if _leaks_local_path(handoff):
        raise SystemExit(f"{label} email search handoff leaked local path: {handoff}")
    external_attempted = bool(handoff.get("external_attempted"))
    if metadata.get("external_attempted", False) is not external_attempted:
        raise SystemExit(f"{label} email search external-attempt parity failed: {metadata}")
    boundaries = handoff.get("boundaries")
    if not isinstance(boundaries, dict):
        raise SystemExit(f"{label} email search handoff missed boundaries: {handoff}")
    expected = {
        "calls_model": False,
        "calls_external_service": external_attempted,
        "executes_tools": False,
        "reads_personal_data": True,
        "reads_private_data": True,
        "executes_side_effect": False,
        "external_side_effect": False,
        "writes_files": False,
        "writes_database": False,
        "writes_memory": False,
        "writes_notes": False,
        "queues_approval": False,
        "requires_approval": False,
        "controls_computer": False,
        **NO_AUTHORITY_FLAGS,
    }
    for key, value in expected.items():
        if boundaries.get(key) is not value:
            raise SystemExit(f"{label} email search boundary {key} wrong: {handoff}")
        if metadata.get(key, value) is not value:
            raise SystemExit(f"{label} email search flat boundary {key} wrong: {metadata}")
    return handoff


def assert_email_body_handoff(metadata: dict, label: str, *, status: str) -> dict:
    handoff = metadata.get("email_body_handoff")
    if not isinstance(handoff, dict):
        raise SystemExit(f"{label} missed email body handoff: {metadata}")
    expected_next = "read email body" if handoff.get("retry_safe") else "search emails"
    assert_command_handoff_contract(
        metadata,
        handoff,
        label,
        prefix="email_body",
        next_safe_command=expected_next,
    )
    if handoff.get("source") != "read_email_body" or handoff.get("status") != status:
        raise SystemExit(f"{label} email body handoff source/status wrong: {handoff}")
    for key in [
        "ready_for_operator",
        "state_changed",
        "changed",
        "content_in_handoff",
        "authorizes_execution",
        "authorizes_completion_claim",
        "approval_granted",
    ]:
        if handoff.get(key) != metadata.get(key):
            raise SystemExit(f"{label} email body handoff should mirror {key}: {handoff} vs {metadata}")
    if handoff.get("limit") != metadata.get("limit"):
        raise SystemExit(f"{label} email body handoff limit parity failed: {metadata}")
    if handoff.get("criterion") != metadata.get("criterion", handoff.get("criterion")):
        raise SystemExit(f"{label} email body criterion parity failed: {metadata}")
    if handoff.get("selected_index") != metadata.get("selected_index", handoff.get("selected_index")):
        raise SystemExit(f"{label} email body selected-index parity failed: {metadata}")
    if handoff.get("body_chars") != metadata.get("body_chars", handoff.get("body_chars")):
        raise SystemExit(f"{label} email body length parity failed: {metadata}")
    if handoff.get("body_in_metadata") is not False:
        raise SystemExit(f"{label} email body handoff should not carry body content: {handoff}")
    if metadata.get("email_body_body_in_metadata") is not False:
        raise SystemExit(f"{label} email body metadata alias should be false: {metadata}")
    if not isinstance(handoff.get("message"), dict):
        raise SystemExit(f"{label} email body handoff missed message preview dict: {handoff}")
    if _leaks_local_path(handoff):
        raise SystemExit(f"{label} email body handoff leaked local path: {handoff}")
    external_attempted = bool(handoff.get("external_attempted"))
    if metadata.get("external_attempted", False) is not external_attempted:
        raise SystemExit(f"{label} email body external-attempt parity failed: {metadata}")
    boundaries = handoff.get("boundaries")
    if not isinstance(boundaries, dict):
        raise SystemExit(f"{label} email body handoff missed boundaries: {handoff}")
    expected = {
        "calls_model": False,
        "calls_external_service": external_attempted,
        "executes_tools": False,
        "reads_personal_data": True,
        "reads_private_data": True,
        "executes_side_effect": False,
        "external_side_effect": False,
        "writes_files": False,
        "writes_database": False,
        "writes_memory": False,
        "writes_notes": False,
        "queues_approval": False,
        "requires_approval": False,
        "controls_computer": False,
        **NO_AUTHORITY_FLAGS,
    }
    for key, value in expected.items():
        if boundaries.get(key) is not value:
            raise SystemExit(f"{label} email body boundary {key} wrong: {handoff}")
        if metadata.get(key, value) is not value:
            raise SystemExit(f"{label} email body flat boundary {key} wrong: {metadata}")
    return handoff


def test_requires_credentials() -> None:
    with mock.patch.dict(os.environ, {"GMAIL_ADDRESS": "", "GMAIL_APP_PASSWORD": ""}, clear=False):
        out = _tools()["send_email"].handler({"to": "a@example.com", "body": "hi"})
    if out.ok or "GMAIL" not in out.output:
        raise SystemExit(f"send_email should require credentials: {out.output}")
    handoff = assert_email_send_handoff(out.metadata, "send_email missing credentials", status="unavailable")
    if handoff.get("reason") != "missing_credentials" or handoff.get("external_attempted") or handoff.get("send_attempted"):
        raise SystemExit(f"send_email missing-credentials handoff should be inert unavailable state: {handoff}")


def test_path_shaped_credentials_are_rejected_locally() -> None:
    raw_address = "/var/folders/zc/gmail-address"
    raw_password = "/tmp/gmail-password"
    tools = _tools()
    with mock.patch.dict(os.environ, {"GMAIL_ADDRESS": raw_address, "GMAIL_APP_PASSWORD": raw_password}, clear=False), \
         mock.patch.object(ec.smtplib, "SMTP_SSL", side_effect=AssertionError("SMTP should not start")), \
         mock.patch.object(ec.imaplib, "IMAP4_SSL", side_effect=AssertionError("IMAP should not start")):
        cases = [
            tools["send_email"].handler({"to": "a@example.com", "body": "hi"}),
            tools["read_emails"].handler({"limit": "/var/folders/zc/email-limit"}),
            tools["search_emails"].handler({"query": "invoice"}),
            tools["read_email_body"].handler({"query": "invoice"}),
        ]
    for out in cases:
        if out.ok or "values hidden" not in out.output:
            raise SystemExit(f"path-shaped Gmail credentials should be rejected locally: {out.output}")
        if raw_address in out.output or raw_password in out.output or raw_address in str(out.metadata) or raw_password in str(out.metadata):
            raise SystemExit(f"path-shaped Gmail credential rejection leaked raw env values: {out.output} {out.metadata}")
        if out.metadata.get("reason") != "invalid_credentials":
            raise SystemExit(f"path-shaped Gmail credentials should expose invalid reason metadata only: {out.metadata}")
        if out.metadata.get("gmail_address_valid") is not False or out.metadata.get("gmail_app_password_valid") is not False:
            raise SystemExit(f"path-shaped Gmail credential metadata should mark both values invalid: {out.metadata}")
        if out.metadata.get("external_attempted") is not False or out.metadata.get("executes_side_effect"):
            raise SystemExit(f"path-shaped Gmail credential rejection must not attempt external work: {out.metadata}")
    for name, out in zip(
        ("read_emails", "search_emails", "read_email_body"),
        cases[1:],
    ):
        assert_personal_read_recovery(out, f"{name} invalid credentials")
    send_handoff = assert_email_send_handoff(cases[0].metadata, "send_email invalid credentials", status="unavailable")
    if send_handoff.get("reason") != "invalid_credentials" or send_handoff.get("external_attempted") or send_handoff.get("send_attempted"):
        raise SystemExit(f"path-shaped Gmail credential handoff should be inert unavailable state: {send_handoff}")
    search_handoff = assert_email_search_handoff(cases[2].metadata, "search_emails invalid credentials", status="unavailable")
    if search_handoff.get("reason") != "invalid_credentials" or search_handoff.get("external_attempted"):
        raise SystemExit(f"path-shaped Gmail search credential handoff should be inert unavailable state: {search_handoff}")


def test_requires_recipient_and_body() -> None:
    with mock.patch.dict(os.environ, {"GMAIL_ADDRESS": "me@example.com", "GMAIL_APP_PASSWORD": "x" * 16}, clear=False):
        tools = _tools()
        missing_to = tools["send_email"].handler({"subject": "Hello", "body": "hi"})
        if missing_to.ok:
            raise SystemExit("send_email accepted missing recipient")
        if missing_to.metadata.get("subject") != "Hello" or missing_to.metadata.get("body_chars") != 2:
            raise SystemExit(f"send_email missing-recipient metadata wrong: {missing_to.metadata}")
        if missing_to.metadata.get("executes_side_effect") or missing_to.metadata.get("external_side_effect"):
            raise SystemExit(f"send_email validation must not claim side effects: {missing_to.metadata}")
        missing_to_handoff = assert_email_send_handoff(missing_to.metadata, "send_email missing recipient", status="refused")
        if missing_to_handoff.get("reason") != "missing_recipient" or missing_to_handoff.get("external_attempted"):
            raise SystemExit(f"send_email missing-recipient handoff should be inert refusal: {missing_to_handoff}")
        missing_body = tools["send_email"].handler({"to": "a@example.com", "subject": "Hello"})
        if missing_body.ok:
            raise SystemExit("send_email accepted missing body")
        if missing_body.metadata.get("to") != "a@example.com" or missing_body.metadata.get("subject") != "Hello" or missing_body.metadata.get("body_chars") != 0:
            raise SystemExit(f"send_email missing-body metadata wrong: {missing_body.metadata}")
        missing_body_handoff = assert_email_send_handoff(missing_body.metadata, "send_email missing body", status="refused")
        if missing_body_handoff.get("reason") != "missing_body" or missing_body_handoff.get("external_attempted"):
            raise SystemExit(f"send_email missing-body handoff should be inert refusal: {missing_body_handoff}")


def test_send_redacts_path_shaped_recipient_metadata() -> None:
    with mock.patch.dict(os.environ, {"GMAIL_ADDRESS": "me@example.com", "GMAIL_APP_PASSWORD": "x" * 16}, clear=False):
        out = _tools()["send_email"].handler({
            "to": "/\x55sers/example/Desktop/Claude code/email-target",
            "subject": "Hello near /tmp/email-subject",
        })
    if out.ok or out.metadata.get("to") != "<local-path>" or out.metadata.get("subject") != "Hello near <local-path>":
        raise SystemExit(f"send_email should redact path-shaped validation metadata: {out.metadata}")
    handoff = assert_email_send_handoff(out.metadata, "send_email path-shaped recipient", status="refused")
    if handoff.get("to") != "<local-path>" or handoff.get("subject") != "Hello near <local-path>" or handoff.get("reason") != "missing_body":
        raise SystemExit(f"send_email path-shaped handoff should redact recipient and subject: {handoff}")
    if _leaks_local_path(out.output) or _leaks_local_path(out.metadata):
        raise SystemExit(f"send_email validation leaked local paths: {out.output} {out.metadata}")


def test_successful_send_is_mocked() -> None:
    sent = {}

    class FakeSMTP:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def login(self, addr, pw):
            sent["login"] = addr

        def sendmail(self, frm, to, msg):
            sent["to"] = to

    with mock.patch.dict(os.environ, {"GMAIL_ADDRESS": "me@example.com", "GMAIL_APP_PASSWORD": "x" * 16}, clear=False), \
         mock.patch.object(ec.smtplib, "SMTP_SSL", FakeSMTP):
        out = _tools()["send_email"].handler({"to": "alex@example.com", "subject": "Hi", "body": "hello"})
    if not out.ok or sent.get("to") != ["alex@example.com"]:
        raise SystemExit(f"mocked send failed: {out.output} {sent}")
    if not out.metadata.get("executes_side_effect"):
        raise SystemExit("send_email must be marked as a side effect")
    handoff = assert_email_send_handoff(out.metadata, "send_email mocked send", status="sent")
    if handoff.get("external_attempted") is not True or handoff.get("send_attempted") is not True:
        raise SystemExit(f"send_email success handoff should preserve send attempt: {handoff}")
    if handoff.get("outcome_known") is not True or handoff.get("retry_safe") is not False:
        raise SystemExit(f"successful send should be known and not repeatable: {handoff}")


def test_send_error_preserves_safe_metadata() -> None:
    class BoomSMTP:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def login(self, addr, pw):
            raise RuntimeError("smtp down")

    with mock.patch.dict(os.environ, {"GMAIL_ADDRESS": "me@example.com", "GMAIL_APP_PASSWORD": "x" * 16}, clear=False), \
         mock.patch.object(ec.smtplib, "SMTP_SSL", BoomSMTP):
        out = _tools()["send_email"].handler({"to": "alex@example.com", "subject": "Hi", "body": "hello"})
    if out.ok or "Gmail could not send" not in out.output:
        raise SystemExit(f"send_email SMTP error should be friendly: {out.output}")
    if "smtp down" in out.output or "Gmail error" in out.output:
        raise SystemExit(f"send_email SMTP error should not leak raw exceptions: {out.output}")
    if out.metadata.get("to") != "alex@example.com" or out.metadata.get("subject") != "Hi" or out.metadata.get("body_chars") != 5:
        raise SystemExit(f"send_email error metadata should preserve safe bounded diagnostics: {out.metadata}")
    if not out.metadata.get("external_attempted") or out.metadata.get("send_attempted"):
        raise SystemExit(f"login failure should preserve external attempt without send attempt: {out.metadata}")
    if out.metadata.get("executes_side_effect") or out.metadata.get("external_side_effect"):
        raise SystemExit(f"login failure should not claim a send side effect: {out.metadata}")
    if out.metadata.get("exception_type") != "RuntimeError":
        raise SystemExit(f"send_email SMTP error should preserve bounded diagnostic metadata: {out.metadata}")
    handoff = assert_email_send_handoff(out.metadata, "send_email login failure", status="failed")
    if handoff.get("reason") != "send_error" or handoff.get("external_attempted") is not True or handoff.get("send_attempted"):
        raise SystemExit(f"send_email login failure handoff should preserve external-only attempt: {handoff}")
    if handoff.get("boundaries", {}).get("external_side_effect") or handoff.get("boundaries", {}).get("executes_side_effect"):
        raise SystemExit(f"send_email login failure handoff should not claim send side effects: {handoff}")
    if out.metadata.get("outcome_known") is not True or out.metadata.get("retry_safe") is not True:
        raise SystemExit(f"pre-send transient login failure should be known and retry-safe: {out.metadata}")
    assert_known_not_sent_recovery(out, "send_email login failure")


def test_send_auth_error_names_the_fix() -> None:
    class BoomAuthSMTP:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def login(self, addr, pw):
            raise ec.smtplib.SMTPAuthenticationError(535, b"Username and Password not accepted")

    with mock.patch.dict(os.environ, {"GMAIL_ADDRESS": "me@example.com", "GMAIL_APP_PASSWORD": "x" * 16}, clear=False), \
         mock.patch.object(ec.smtplib, "SMTP_SSL", BoomAuthSMTP):
        out = _tools()["send_email"].handler({"to": "alex@example.com", "subject": "Hi", "body": "hello"})
    if out.ok:
        raise SystemExit("send_email auth failure should not report ok")
    if "app password" not in out.output.lower() or "retrying will not help" not in out.output.lower():
        raise SystemExit(f"send_email auth error should name the fix, not the generic retry message: {out.output}")
    if "535" in out.output or "not accepted" in out.output.lower():
        raise SystemExit(f"send_email auth error should not leak the raw SMTP error text: {out.output}")
    handoff = assert_email_send_handoff(out.metadata, "send_email auth failure", status="failed")
    if out.metadata.get("outcome_known") is not True or out.metadata.get("retry_safe") is not False:
        raise SystemExit(f"auth failure should be known but blocked until credential repair: {out.metadata}")
    if handoff.get("outcome_known") is not True or handoff.get("retry_safe") is not False:
        raise SystemExit(f"auth failure handoff retry truth wrong: {handoff}")


def test_read_auth_error_names_the_fix() -> None:
    class BoomAuthIMAP:
        def __init__(self, *a, **k):
            pass

        def login(self, addr, pw):
            raise ec.imaplib.IMAP4.error("AUTHENTICATIONFAILED")

        def logout(self):
            pass

    with mock.patch.dict(os.environ, {"GMAIL_ADDRESS": "me@example.com", "GMAIL_APP_PASSWORD": "x" * 16}, clear=False), \
         mock.patch.object(ec.imaplib, "IMAP4_SSL", BoomAuthIMAP):
        out = _tools()["read_emails"].handler({})
    if out.ok:
        raise SystemExit("read_emails auth failure should not report ok")
    if "app password" not in out.output.lower() or "retrying will not help" not in out.output.lower():
        raise SystemExit(f"read_emails auth error should name the fix, not the generic retry message: {out.output}")


def test_generic_email_error_keeps_retry_message() -> None:
    """Non-auth failures should NOT claim the app password is wrong -- that would misdirect the fix."""
    class BoomNetworkSMTP:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def login(self, addr, pw):
            raise TimeoutError("connection timed out")

    with mock.patch.dict(os.environ, {"GMAIL_ADDRESS": "me@example.com", "GMAIL_APP_PASSWORD": "x" * 16}, clear=False), \
         mock.patch.object(ec.smtplib, "SMTP_SSL", BoomNetworkSMTP):
        out = _tools()["send_email"].handler({"to": "alex@example.com", "subject": "Hi", "body": "hello"})
    if out.ok:
        raise SystemExit("send_email network failure should not report ok")
    if "app password" in out.output.lower() or "retrying will not help" in out.output.lower():
        raise SystemExit(f"send_email transient error should not claim a permanent auth failure: {out.output}")
    if "Gmail could not send" not in out.output:
        raise SystemExit(f"send_email transient error should keep the generic retry message: {out.output}")
    expected_recovery = [
        "network access to Gmail",
        "GMAIL_ADDRESS",
        "GMAIL_APP_PASSWORD",
        "setup check",
        "retry",
    ]
    for expected in expected_recovery:
        if expected not in out.output:
            raise SystemExit(f"send_email transient error missed recovery step {expected!r}: {out.output}")
    handoff = assert_email_send_handoff(out.metadata, "send_email pre-send network failure", status="failed")
    if out.metadata.get("retry_safe") is not True or out.metadata.get("outcome_known") is not True:
        raise SystemExit(f"pre-send network failure should remain retry-safe and known: {out.metadata}")
    if handoff.get("retry_safe") is not True or handoff.get("outcome_known") is not True:
        raise SystemExit(f"pre-send network handoff truth wrong: {handoff}")


def test_sendmail_error_preserves_attempt_metadata() -> None:
    accepted = {"payload": False}

    class BoomSendSMTP:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def login(self, addr, pw):
            return None

        def sendmail(self, frm, to, msg):
            accepted["payload"] = True
            raise ec.smtplib.SMTPServerDisconnected("connection lost after DATA")

    with mock.patch.dict(os.environ, {"GMAIL_ADDRESS": "me@example.com", "GMAIL_APP_PASSWORD": "x" * 16}, clear=False), \
         mock.patch.object(ec.smtplib, "SMTP_SSL", BoomSendSMTP):
        out = _tools()["send_email"].handler({"to": "alex@example.com", "subject": "Hi", "body": "hello"})
    if out.ok or "outcome is unknown" not in out.output:
        raise SystemExit(f"send_email sendmail error should report uncertainty: {out.output}")
    for expected in ("may already have been sent", "do not retry blindly", "Gmail Sent"):
        if expected not in out.output:
            raise SystemExit(f"send_email uncertain outcome missed {expected!r}: {out.output}")
    if "then retry" in out.output.lower():
        raise SystemExit(f"send_email uncertain outcome encouraged a blind retry: {out.output}")
    if "connection lost after DATA" in out.output or "Gmail error" in out.output:
        raise SystemExit(f"send_email sendmail error should not leak raw exceptions: {out.output}")
    if accepted["payload"] is not True:
        raise SystemExit("sendmail uncertainty fixture did not reach the post-DATA boundary")
    if not out.metadata.get("external_attempted") or not out.metadata.get("send_attempted"):
        raise SystemExit(f"sendmail failure should preserve attempted send metadata: {out.metadata}")
    if not out.metadata.get("executes_side_effect") or not out.metadata.get("external_side_effect"):
        raise SystemExit(f"sendmail failure should mark the attempted external side effect: {out.metadata}")
    if out.metadata.get("exception_type") != "SMTPServerDisconnected":
        raise SystemExit(f"send_email sendmail error should preserve bounded diagnostic metadata: {out.metadata}")
    if out.metadata.get("retry_safe") is not False or out.metadata.get("outcome_known") is not False:
        raise SystemExit(f"sendmail uncertainty should block retry and completion claims: {out.metadata}")
    if (
        out.metadata.get("outcome_unknown") is not True
        or out.metadata.get("execution_outcome_unknown") is not True
        or out.metadata.get("side_effect_possible") is not True
        or out.metadata.get("automatic_retry_allowed") is not False
        or out.metadata.get("authorizes_retry") is not False
        or out.metadata.get("next_command") != "setup check"
        or out.metadata.get("recovery_commands") != ["setup check"]
        or out.metadata.get("recovery_guidance") != {
            "version": 1,
            "action": out.output,
            "commands": ["setup check"],
        }
    ):
        raise SystemExit(f"sendmail uncertainty lacks canonical recovery declaration: {out.metadata}")
    handoff = assert_email_send_handoff(
        out.metadata,
        "send_email sendmail failure",
        status="outcome_unknown",
    )
    if handoff.get("reason") != "send_outcome_unknown" or handoff.get("external_attempted") is not True or handoff.get("send_attempted") is not True:
        raise SystemExit(f"send_email sendmail failure handoff should preserve attempted send: {handoff}")
    if handoff.get("retry_safe") is not False or handoff.get("outcome_known") is not False:
        raise SystemExit(f"sendmail uncertainty handoff should require manual verification: {handoff}")


def test_send_resolves_contact_names_before_smtp() -> None:
    sent = {}

    class FakeSMTP:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def login(self, addr, pw):
            sent["login"] = addr

        def sendmail(self, frm, to, msg):
            sent["to"] = to

    with mock.patch.dict(os.environ, {"GMAIL_ADDRESS": "me@example.com", "GMAIL_APP_PASSWORD": "x" * 16}, clear=False), \
         mock.patch.object(ec.smtplib, "SMTP_SSL", FakeSMTP), \
         mock.patch.object(ec.contacts_connector, "resolve_contact", return_value=[
             ec.contacts_connector.ContactMatch("Fixture Example", email="fixture@example.com")
         ]):
        out = _tools()["send_email"].handler({"to": "fixture", "subject": "Hi", "body": "hello"})
    if not out.ok or sent.get("to") != ["fixture@example.com"]:
        raise SystemExit(f"send_email should resolve one contact before SMTP: {out.output} {sent}")
    if out.metadata.get("to") != "fixture" or out.metadata.get("resolved_to") != "fixture@example.com":
        raise SystemExit(f"send_email should preserve original and resolved recipients: {out.metadata}")
    handoff = assert_email_send_handoff(out.metadata, "send_email resolved contact", status="sent")
    if handoff.get("contact_resolution_status") != "resolved" or not handoff.get("contact_lookup_attempted"):
        raise SystemExit(f"send_email resolved contact handoff wrong: {handoff}")


def test_send_ambiguous_contact_refuses_without_smtp() -> None:
    with mock.patch.dict(os.environ, {"GMAIL_ADDRESS": "me@example.com", "GMAIL_APP_PASSWORD": "x" * 16}, clear=False), \
         mock.patch.object(ec.smtplib, "SMTP_SSL", side_effect=AssertionError("SMTP should not start")), \
         mock.patch.object(ec.contacts_connector, "resolve_contact", return_value=[
             ec.contacts_connector.ContactMatch("Fixture Example", email="fixture@example.com"),
             ec.contacts_connector.ContactMatch("Fixture Kim", email="fixture@example.com"),
         ]):
        out = _tools()["send_email"].handler({"to": "fixture", "subject": "Hi", "body": "hello"})
    if out.ok or "Fixture Example" not in out.output or "Fixture Kim" not in out.output:
        raise SystemExit(f"send_email ambiguous contact should ask which contact: {out.output}")
    handoff = assert_email_send_handoff(out.metadata, "send_email ambiguous contact", status="refused")
    if handoff.get("contact_resolution_status") != "ambiguous" or handoff.get("send_attempted"):
        raise SystemExit(f"send_email ambiguous contact handoff wrong: {handoff}")


def test_send_missing_contact_refuses_without_smtp() -> None:
    with mock.patch.dict(os.environ, {"GMAIL_ADDRESS": "me@example.com", "GMAIL_APP_PASSWORD": "x" * 16}, clear=False), \
         mock.patch.object(ec.smtplib, "SMTP_SSL", side_effect=AssertionError("SMTP should not start")), \
         mock.patch.object(ec.contacts_connector, "resolve_contact", return_value=[]):
        out = _tools()["send_email"].handler({"to": "fixture", "subject": "Hi", "body": "hello"})
    if out.ok or "couldn't find" not in out.output.lower():
        raise SystemExit(f"send_email missing contact should refuse without SMTP: {out.output}")
    handoff = assert_email_send_handoff(out.metadata, "send_email missing contact", status="refused")
    if handoff.get("contact_resolution_status") != "not_found" or handoff.get("send_attempted"):
        raise SystemExit(f"send_email missing contact handoff wrong: {handoff}")


def test_send_malformed_contact_resolution_defaults_closed() -> None:
    malformed_metadata = {
        "original_to": "fixture",
        "contact_lookup_attempted": "true",
        "contact_resolution_status": "not_found",
        "contact_match_count": 0,
        "contact_candidates": [],
    }
    with mock.patch.dict(os.environ, {"GMAIL_ADDRESS": "me@example.com", "GMAIL_APP_PASSWORD": "x" * 16}, clear=False), \
         mock.patch.object(ec.smtplib, "SMTP_SSL", side_effect=AssertionError("SMTP should not start")), \
         mock.patch.object(ec, "_resolve_send_recipient", return_value=("", malformed_metadata, "I couldn't find that contact.")):
        out = _tools()["send_email"].handler({"to": "fixture", "subject": "Hi", "body": "hello"})
    if out.ok or out.metadata.get("send_attempted"):
        raise SystemExit(f"send_email malformed contact metadata should refuse without SMTP: {out.output} {out.metadata}")
    handoff = assert_email_send_handoff(out.metadata, "send_email malformed contact metadata", status="refused")
    if out.metadata.get("contact_lookup_attempted") is not False or handoff.get("contact_lookup_attempted") is not False:
        raise SystemExit(f"send_email should normalize malformed contact lookup flags closed: {out.metadata} {handoff}")
    if out.metadata.get("reads_personal_data") is not False:
        raise SystemExit(f"send_email malformed contact metadata should not claim personal-data reads: {out.metadata}")
    if handoff.get("boundaries", {}).get("reads_personal_data") is not False:
        raise SystemExit(f"send_email malformed contact metadata handoff boundary should stay closed: {handoff}")


def test_send_explicit_email_skips_contact_resolution() -> None:
    sent = {}

    class FakeSMTP:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def login(self, addr, pw):
            sent["login"] = addr

        def sendmail(self, frm, to, msg):
            sent["to"] = to

    with mock.patch.dict(os.environ, {"GMAIL_ADDRESS": "me@example.com", "GMAIL_APP_PASSWORD": "x" * 16}, clear=False), \
         mock.patch.object(ec.smtplib, "SMTP_SSL", FakeSMTP), \
         mock.patch.object(ec.contacts_connector, "resolve_contact", side_effect=AssertionError("explicit email should not resolve contacts")):
        out = _tools()["send_email"].handler({"to": "alex@example.com", "subject": "Hi", "body": "hello"})
    if not out.ok or sent.get("to") != ["alex@example.com"]:
        raise SystemExit(f"send_email explicit email should send as-is: {out.output} {sent}")
    handoff = assert_email_send_handoff(out.metadata, "send_email explicit email", status="sent")
    if handoff.get("contact_resolution_status") != "skipped" or handoff.get("contact_lookup_attempted"):
        raise SystemExit(f"send_email explicit email handoff wrong: {handoff}")


def test_read_requires_credentials() -> None:
    with mock.patch.dict(os.environ, {"GMAIL_ADDRESS": "", "GMAIL_APP_PASSWORD": ""}, clear=False):
        out = _tools()["read_emails"].handler({"limit": "bad"})
    if out.ok or "GMAIL" not in out.output:
        raise SystemExit(f"read_emails should require credentials: {out.output}")
    if out.metadata.get("limit") != 5 or out.metadata.get("raw_limit") != "bad":
        raise SystemExit(f"read_emails credential metadata should preserve sanitized raw limit: {out.metadata}")
    handoff = assert_email_read_handoff(out.metadata, "read_emails missing credentials", status="unavailable")
    if handoff.get("reason") != "missing_credentials" or handoff.get("external_attempted"):
        raise SystemExit(f"read_emails missing credentials should be inert unavailable state: {handoff}")
    with mock.patch.dict(os.environ, {"GMAIL_ADDRESS": "", "GMAIL_APP_PASSWORD": ""}, clear=False):
        path_out = _tools()["read_emails"].handler({"limit": "/\x55sers/example/private/email-limit"})
    if path_out.ok or path_out.metadata.get("limit") != 5 or path_out.metadata.get("raw_limit") != "<local-path>":
        raise SystemExit(f"read_emails leaked local path in raw limit metadata: {path_out.metadata}")
    path_handoff = assert_email_read_handoff(path_out.metadata, "read_emails path limit credentials", status="unavailable")
    if path_handoff.get("reason") != "missing_credentials":
        raise SystemExit(f"read_emails path-limit missing credentials reason wrong: {path_handoff}")


def test_read_emails_mocked() -> None:
    calls = []

    class FakeIMAP:
        def __init__(self, *a, **k):
            pass

        def login(self, *a):
            return ("OK", [])

        def select(self, *a, **k):
            return ("OK", [])

        def search(self, charset, criterion):
            calls.append(criterion)
            return ("OK", [b"1 2 3"])

        def fetch(self, mid, spec):
            raw = b"From: Alex </\x55sers/example/Desktop/Claude code/alex.txt>\r\nSubject: Hello near /private/tmp/email-subject and /var/folders/zc/email-cache and /tmp/email-scratch\r\n\r\n"
            return ("OK", [(b"1", raw)])

        def logout(self):
            return ("BYE", [])

    with mock.patch.dict(os.environ, {"GMAIL_ADDRESS": "me@example.com", "GMAIL_APP_PASSWORD": "x" * 16}, clear=False), \
         mock.patch.object(ec.imaplib, "IMAP4_SSL", FakeIMAP):
        out = _tools()["read_emails"].handler({"limit": "bad"})
    if not out.ok or "Hello near" not in out.output or "Alex" not in out.output:
        raise SystemExit(f"read_emails mocked output wrong: {out.output}")
    if any(fragment in out.output for fragment in ["/\x55sers/", "/private/", "/var/folders/", "/tmp/"]) or "<local-path>" not in out.output:
        raise SystemExit(f"read_emails should redact path-shaped headers: {out.output}")
    if out.metadata.get("executes_side_effect"):
        raise SystemExit("read_emails must not be a side effect")
    if out.metadata.get("limit") != 5 or out.metadata.get("raw_limit") != "bad":
        raise SystemExit(f"read_emails should preserve sanitized raw limit metadata: {out.metadata}")
    handoff = assert_email_read_handoff(out.metadata, "read_emails mocked", status="ok")
    if handoff.get("count") != 3 or handoff.get("row_count") != 3:
        raise SystemExit(f"read_emails handoff should preserve header row count: {handoff}")
    if "Hello near" not in str(handoff.get("rows")) or "Alex" not in str(handoff.get("rows")):
        raise SystemExit(f"read_emails handoff should preserve sanitized header previews: {handoff}")
    if calls[-1] != "ALL" or handoff.get("unread_only") is not False:
        raise SystemExit(f"read_emails default should search all messages: {calls} {handoff}")
    with mock.patch.dict(os.environ, {"GMAIL_ADDRESS": "me@example.com", "GMAIL_APP_PASSWORD": "x" * 16}, clear=False), \
         mock.patch.object(ec.imaplib, "IMAP4_SSL", FakeIMAP):
        malformed_unread = _tools()["read_emails"].handler({"limit": 2, "unread": "false"})
    malformed_handoff = assert_email_read_handoff(
        malformed_unread.metadata,
        "read_emails malformed unread flag",
        status="ok",
    )
    if calls[-1] != "ALL" or malformed_handoff.get("unread_only") is not False:
        raise SystemExit(f"read_emails malformed unread flag should not narrow to UNSEEN: {calls} {malformed_handoff}")
    with mock.patch.dict(os.environ, {"GMAIL_ADDRESS": "me@example.com", "GMAIL_APP_PASSWORD": "x" * 16}, clear=False), \
         mock.patch.object(ec.imaplib, "IMAP4_SSL", FakeIMAP):
        bool_out = _tools()["read_emails"].handler({"limit": False})
    if not bool_out.ok or bool_out.metadata.get("limit") != 5 or bool_out.metadata.get("raw_limit") != "False":
        raise SystemExit(f"read_emails should treat boolean limits as malformed defaults: {bool_out.metadata}")
    assert_email_read_handoff(bool_out.metadata, "read_emails boolean limit", status="ok")


def test_read_emails_empty_handoff() -> None:
    class EmptyIMAP:
        def __init__(self, *a, **k):
            pass

        def login(self, *a):
            return ("OK", [])

        def select(self, *a, **k):
            return ("OK", [])

        def search(self, charset, criterion):
            return ("OK", [b""])

        def logout(self):
            return ("BYE", [])

    with mock.patch.dict(os.environ, {"GMAIL_ADDRESS": "me@example.com", "GMAIL_APP_PASSWORD": "x" * 16}, clear=False), \
         mock.patch.object(ec.imaplib, "IMAP4_SSL", EmptyIMAP):
        out = _tools()["read_emails"].handler({"limit": 3, "unread": True})
    if not out.ok or "No unread emails" not in out.output:
        raise SystemExit(f"read_emails empty output wrong: {out.output}")
    handoff = assert_email_read_handoff(out.metadata, "read_emails empty", status="empty")
    if handoff.get("count") != 0 or handoff.get("row_count") != 0 or handoff.get("unread_only") is not True:
        raise SystemExit(f"read_emails empty handoff should preserve empty unread state: {handoff}")


def test_read_email_failures_are_clean() -> None:
    class BoomIMAP:
        def __init__(self, *a, **k):
            raise RuntimeError("imap connect down")

    with mock.patch.dict(os.environ, {"GMAIL_ADDRESS": "me@example.com", "GMAIL_APP_PASSWORD": "x" * 16}, clear=False), \
         mock.patch.object(ec.imaplib, "IMAP4_SSL", BoomIMAP):
        out = _tools()["read_emails"].handler({"limit": "bad"})
    if out.ok or "Gmail could not read" not in out.output:
        raise SystemExit(f"read_emails IMAP error should be friendly: {out.output}")
    assert_personal_read_recovery(out, "read_emails IMAP failure")
    if "imap connect down" in out.output or "Gmail IMAP error" in out.output:
        raise SystemExit(f"read_emails IMAP error should not leak raw exceptions: {out.output}")
    for expected in ["network access to Gmail", "GMAIL_ADDRESS", "GMAIL_APP_PASSWORD", "setup check", "retry"]:
        if expected not in out.output:
            raise SystemExit(f"read_emails IMAP error missed recovery step {expected!r}: {out.output}")
    if out.metadata.get("exception_type") != "RuntimeError" or out.metadata.get("raw_limit") != "bad":
        raise SystemExit(f"read_emails IMAP error should preserve bounded metadata: {out.metadata}")
    handoff = assert_email_read_handoff(out.metadata, "read_emails IMAP failure", status="failed")
    if handoff.get("reason") != "read_error" or handoff.get("retry_safe") is not True or handoff.get("external_attempted") is not True:
        raise SystemExit(f"read_emails IMAP failure handoff should preserve retryable external attempt: {handoff}")


def test_search_emails_mocked_and_personal_data_gated() -> None:
    calls = {}

    class FakeIMAP:
        def __init__(self, *a, **k):
            pass

        def login(self, *a):
            return ("OK", [])

        def select(self, *a, **k):
            return ("OK", [])

        def search(self, charset, criterion):
            calls["criterion"] = criterion
            return ("OK", [b"4 5"])

        def fetch(self, mid, spec):
            raw = b"From: Billing </\x55sers/example/Desktop/Claude code/billing.txt>\r\nSubject: Invoice near /private/tmp/email-search and /var/folders/zc/email-search and /tmp/email-search\r\nDate: Mon, 15 Jun 2026 09:00:00 +0000\r\n\r\n"
            return ("OK", [(b"1", raw)])

        def logout(self):
            return ("BYE", [])

    tools = _tools()
    if tools["search_emails"].risk != RiskLevel.PERSONAL_DATA:
        raise SystemExit("search_emails must be PERSONAL_DATA gated")
    with mock.patch.dict(os.environ, {"GMAIL_ADDRESS": "me@example.com", "GMAIL_APP_PASSWORD": "x" * 16}, clear=False), \
         mock.patch.object(ec.imaplib, "IMAP4_SSL", FakeIMAP):
        out = tools["search_emails"].handler({"query": "invoice", "limit": "bad"})
    if not out.ok or "Invoice near" not in out.output or "Billing" not in out.output:
        raise SystemExit(f"search_emails mocked output wrong: {out.output}")
    if any(fragment in out.output for fragment in ["/\x55sers/", "/private/", "/var/folders/", "/tmp/"]) or "<local-path>" not in out.output:
        raise SystemExit(f"search_emails should redact path-shaped headers: {out.output}")
    if 'TEXT "invoice"' not in calls.get("criterion", ""):
        raise SystemExit(f"search_emails did not use TEXT search criterion: {calls}")
    if not out.metadata.get("reads_private_data") or out.metadata.get("executes_side_effect"):
        raise SystemExit(f"search_emails metadata wrong: {out.metadata}")
    if out.metadata.get("limit") != 10 or out.metadata.get("raw_limit") != "bad":
        raise SystemExit(f"search_emails should preserve sanitized raw limit metadata: {out.metadata}")
    handoff = assert_email_search_handoff(out.metadata, "search_emails mocked", status="ok")
    if handoff.get("count") != 2 or handoff.get("row_count") != 2:
        raise SystemExit(f"search_emails handoff should preserve header row count: {handoff}")
    if "Invoice near" not in str(handoff.get("rows")) or "Billing" not in str(handoff.get("rows")):
        raise SystemExit(f"search_emails handoff should preserve sanitized header previews: {handoff}")
    with mock.patch.dict(os.environ, {"GMAIL_ADDRESS": "me@example.com", "GMAIL_APP_PASSWORD": "x" * 16}, clear=False), \
         mock.patch.object(ec.imaplib, "IMAP4_SSL", FakeIMAP):
        bool_out = tools["search_emails"].handler({"query": "invoice", "limit": True})
    if not bool_out.ok or bool_out.metadata.get("limit") != 10 or bool_out.metadata.get("raw_limit") != "True":
        raise SystemExit(f"search_emails should treat boolean limits as malformed defaults: {bool_out.metadata}")
    if bool_out.metadata.get("executes_side_effect"):
        raise SystemExit(f"search_emails boolean limit should not be a side effect: {bool_out.metadata}")
    assert_email_search_handoff(bool_out.metadata, "search_emails boolean limit", status="ok")
    with mock.patch.dict(os.environ, {"GMAIL_ADDRESS": "me@example.com", "GMAIL_APP_PASSWORD": "x" * 16}, clear=False), \
         mock.patch.object(ec.imaplib, "IMAP4_SSL", FakeIMAP):
        path_out = tools["search_emails"].handler({"query": "invoice", "limit": "/var/folders/zc/jarvis-email-search-limit"})
    if not path_out.ok or path_out.metadata.get("limit") != 10 or path_out.metadata.get("raw_limit") != "<local-path>":
        raise SystemExit(f"search_emails leaked local path in raw limit metadata: {path_out.metadata}")
    assert_email_search_handoff(path_out.metadata, "search_emails path limit", status="ok")
    with mock.patch.dict(os.environ, {"GMAIL_ADDRESS": "me@example.com", "GMAIL_APP_PASSWORD": "x" * 16}, clear=False), \
         mock.patch.object(ec.imaplib, "IMAP4_SSL", FakeIMAP):
        bad_query = tools["search_emails"].handler({"query": "/tmp/email-query"})
    if bad_query.ok or bad_query.metadata.get("reason") != "invalid_search_term" or bad_query.metadata.get("criterion") != "<local-path>":
        raise SystemExit(f"search_emails should reject path-shaped criteria locally: {bad_query.output} {bad_query.metadata}")
    if any(fragment in bad_query.output for fragment in ["/\x55sers/", "/private/", "/var/folders/", "/tmp/"]) or any(fragment in str(bad_query.metadata) for fragment in ["/\x55sers/", "/private/", "/var/folders/", "/tmp/"]):
        raise SystemExit(f"search_emails path-shaped rejection leaked local paths: {bad_query.output} {bad_query.metadata}")
    bad_handoff = assert_email_search_handoff(bad_query.metadata, "search_emails path query", status="refused")
    if bad_handoff.get("reason") != "invalid_search_term" or bad_handoff.get("external_attempted"):
        raise SystemExit(f"search_emails path query handoff should be inert refusal: {bad_handoff}")


def test_unicode_email_search_uses_utf8_literal() -> None:
    calls = []

    class Utf8IMAP:
        def __init__(self, *a, **k):
            pass

        def login(self, *a):
            calls.append("login")
            return ("OK", [])

        def select(self, *a, **k):
            calls.append("select")
            return ("OK", [])

        def search(self, charset, *criteria):
            calls.append(("search", charset, *criteria, getattr(self, "literal", None)))
            return ("OK", [b""])

        def logout(self):
            return ("BYE", [])

    with mock.patch.dict(os.environ, {"GMAIL_ADDRESS": "me@example.com", "GMAIL_APP_PASSWORD": "x" * 16}, clear=False), \
         mock.patch.object(ec.imaplib, "IMAP4_SSL", Utf8IMAP):
        out = _tools()["search_emails"].handler({"query": "영수증"})
    if not out.ok or "No matching emails" not in out.output:
        raise SystemExit(f"UTF-8 email search should complete cleanly: {out.output}")
    expected = [
        "login",
        "select",
        ("search", "UTF-8", b"TEXT", "영수증".encode("utf-8")),
    ]
    if calls != expected:
        raise SystemExit(f"UTF-8 email search must use CHARSET UTF-8 with a literal: {calls}")
    assert_email_search_handoff(out.metadata, "UTF-8 email search", status="empty")


def test_unicode_email_search_preserves_structured_gmail_terms() -> None:
    calls = []

    class StructuredIMAP:
        def __init__(self, *a, **k):
            pass

        def login(self, *a):
            return ("OK", [])

        def select(self, *a, **k):
            return ("OK", [])

        def search(self, charset, *criteria):
            calls.append(("search", charset, *criteria, getattr(self, "literal", None)))
            if len(calls) == 1:
                return ("OK", [b"1 2"])
            if len(calls) == 2:
                return ("OK", [b"2"])
            return ("OK", [b""])

        def logout(self):
            return ("BYE", [])

    with mock.patch.dict(os.environ, {"GMAIL_ADDRESS": "me@example.com", "GMAIL_APP_PASSWORD": "x" * 16}, clear=False), \
         mock.patch.object(ec.imaplib, "IMAP4_SSL", StructuredIMAP):
        out = _tools()["search_emails"].handler({
            "sender": "결제팀",
            "subject": "7월 영수증",
            "query": "카드 결제",
        })
    expected = [
        ("search", "UTF-8", b"FROM", "결제팀".encode("utf-8")),
        ("search", "UTF-8", b"SUBJECT", "7월 영수증".encode("utf-8")),
        ("search", "UTF-8", b"TEXT", "카드 결제".encode("utf-8")),
    ]
    if not out.ok or calls != expected:
        raise SystemExit(f"UTF-8 Gmail search should preserve structured terms: {out.output} {calls}")
    assert_email_search_handoff(out.metadata, "structured UTF-8 Gmail search", status="empty")


def test_unicode_email_body_search_uses_same_utf8_literal_path() -> None:
    calls = []

    class Utf8BodyIMAP:
        def __init__(self, *a, **k):
            pass

        def login(self, *a):
            return ("OK", [])

        def select(self, *a, **k):
            return ("OK", [])

        def search(self, charset, *criteria):
            calls.append(("search", charset, *criteria, getattr(self, "literal", None)))
            return ("OK", [b""])

        def logout(self):
            return ("BYE", [])

    with mock.patch.dict(os.environ, {"GMAIL_ADDRESS": "me@example.com", "GMAIL_APP_PASSWORD": "x" * 16}, clear=False), \
         mock.patch.object(ec.imaplib, "IMAP4_SSL", Utf8BodyIMAP):
        out = _tools()["read_email_body"].handler({"query": "영수증"})
    expected = [("search", "UTF-8", b"TEXT", "영수증".encode("utf-8"))]
    if not out.ok or calls != expected:
        raise SystemExit(f"UTF-8 email body search should use a UTF-8 literal: {out.output} {calls}")
    assert_email_body_handoff(out.metadata, "UTF-8 email body search", status="empty")


def test_unicode_email_search_records_safe_bad_stage() -> None:
    class BadSearchIMAP:
        def __init__(self, *a, **k):
            pass

        def login(self, *a):
            return ("OK", [])

        def select(self, *a, **k):
            return ("OK", [])

        def search(self, charset, *criteria):
            raise ec.imaplib.IMAP4.error("private Gmail protocol detail")

        def logout(self):
            return ("BYE", [])

    with mock.patch.dict(os.environ, {"GMAIL_ADDRESS": "me@example.com", "GMAIL_APP_PASSWORD": "x" * 16}, clear=False), \
         mock.patch.object(ec.imaplib, "IMAP4_SSL", BadSearchIMAP):
        out = _tools()["search_emails"].handler({"query": "영수증"})
    if out.ok or out.metadata.get("imap_operation") != "search" or out.metadata.get("imap_status") != "BAD":
        raise SystemExit(f"UTF-8 Gmail BAD response should preserve only safe stage metadata: {out.metadata}")
    if "private Gmail protocol detail" in out.output or "private Gmail protocol detail" in str(out.metadata):
        raise SystemExit(f"UTF-8 Gmail BAD response leaked raw protocol detail: {out.output} {out.metadata}")
    assert_email_search_handoff(out.metadata, "UTF-8 Gmail BAD stage", status="failed")


def test_search_emails_empty_handoff() -> None:
    class EmptyIMAP:
        def __init__(self, *a, **k):
            pass

        def login(self, *a):
            return ("OK", [])

        def select(self, *a, **k):
            return ("OK", [])

        def search(self, charset, criterion):
            return ("OK", [b""])

        def logout(self):
            return ("BYE", [])

    with mock.patch.dict(os.environ, {"GMAIL_ADDRESS": "me@example.com", "GMAIL_APP_PASSWORD": "x" * 16}, clear=False), \
         mock.patch.object(ec.imaplib, "IMAP4_SSL", EmptyIMAP):
        out = _tools()["search_emails"].handler({"query": "invoice", "limit": 4})
    if not out.ok or "No matching emails" not in out.output:
        raise SystemExit(f"search_emails empty output wrong: {out.output}")
    handoff = assert_email_search_handoff(out.metadata, "search_emails empty", status="empty")
    if handoff.get("count") != 0 or handoff.get("row_count") != 0 or handoff.get("criterion") != 'TEXT "invoice"':
        raise SystemExit(f"search_emails empty handoff should preserve empty criterion state: {handoff}")


def test_read_email_body_empty_handoff() -> None:
    class EmptyIMAP:
        def __init__(self, *a, **k):
            pass

        def login(self, *a):
            return ("OK", [])

        def select(self, *a, **k):
            return ("OK", [])

        def search(self, charset, criterion):
            return ("OK", [b""])

        def logout(self):
            return ("BYE", [])

    with mock.patch.dict(os.environ, {"GMAIL_ADDRESS": "me@example.com", "GMAIL_APP_PASSWORD": "x" * 16}, clear=False), \
         mock.patch.object(ec.imaplib, "IMAP4_SSL", EmptyIMAP):
        out = _tools()["read_email_body"].handler({"query": "invoice", "limit": 4, "index": 2})
    if not out.ok or "No matching emails" not in out.output:
        raise SystemExit(f"read_email_body empty output wrong: {out.output}")
    handoff = assert_email_body_handoff(out.metadata, "read_email_body empty", status="empty")
    if handoff.get("count") != 0 or handoff.get("body_chars") != 0 or handoff.get("criterion") != 'TEXT "invoice"':
        raise SystemExit(f"read_email_body empty handoff should preserve empty criterion state: {handoff}")
    if handoff.get("selected_index") != 2:
        raise SystemExit(f"read_email_body empty handoff should preserve requested selected index: {handoff}")


def test_search_and_body_failures_are_clean() -> None:
    class BoomIMAP:
        def __init__(self, *a, **k):
            pass

        def login(self, *a):
            return ("OK", [])

        def select(self, *a, **k):
            return ("OK", [])

        def search(self, charset, criterion):
            raise RuntimeError("imap search down")

        def logout(self):
            return ("BYE", [])

    tools = _tools()
    with mock.patch.dict(os.environ, {"GMAIL_ADDRESS": "me@example.com", "GMAIL_APP_PASSWORD": "x" * 16}, clear=False), \
         mock.patch.object(ec.imaplib, "IMAP4_SSL", BoomIMAP):
        search_out = tools["search_emails"].handler({"query": "invoice", "limit": "bad"})
    if search_out.ok or "Gmail could not search" not in search_out.output:
        raise SystemExit(f"search_emails IMAP error should be friendly: {search_out.output}")
    assert_personal_read_recovery(search_out, "search_emails IMAP failure")
    if "imap search down" in search_out.output or "Gmail IMAP search error" in search_out.output:
        raise SystemExit(f"search_emails IMAP error should not leak raw exceptions: {search_out.output}")
    for expected in ["network access to Gmail", "GMAIL_ADDRESS", "GMAIL_APP_PASSWORD", "setup check", "retry"]:
        if expected not in search_out.output:
            raise SystemExit(f"search_emails IMAP error missed recovery step {expected!r}: {search_out.output}")
    if search_out.metadata.get("exception_type") != "RuntimeError" or search_out.metadata.get("raw_limit") != "bad":
        raise SystemExit(f"search_emails IMAP error should preserve bounded metadata: {search_out.metadata}")
    if not search_out.metadata.get("reads_private_data") or search_out.metadata.get("executes_side_effect"):
        raise SystemExit(f"search_emails IMAP error metadata should preserve risk boundary: {search_out.metadata}")
    search_handoff = assert_email_search_handoff(search_out.metadata, "search_emails IMAP failure", status="failed")
    if search_handoff.get("reason") != "search_error" or search_handoff.get("retry_safe") is not True or search_handoff.get("external_attempted") is not True:
        raise SystemExit(f"search_emails IMAP failure handoff should preserve retryable external attempt: {search_handoff}")

    with mock.patch.dict(os.environ, {"GMAIL_ADDRESS": "me@example.com", "GMAIL_APP_PASSWORD": "x" * 16}, clear=False), \
         mock.patch.object(ec.imaplib, "IMAP4_SSL", BoomIMAP):
        body_out = tools["read_email_body"].handler({"query": "invoice", "limit": "bad"})
    if body_out.ok or "Gmail could not read that message body" not in body_out.output:
        raise SystemExit(f"read_email_body IMAP error should be friendly: {body_out.output}")
    assert_personal_read_recovery(body_out, "read_email_body IMAP failure")
    if "imap search down" in body_out.output or "Gmail IMAP body error" in body_out.output:
        raise SystemExit(f"read_email_body IMAP error should not leak raw exceptions: {body_out.output}")
    for expected in ["network access to Gmail", "GMAIL_ADDRESS", "GMAIL_APP_PASSWORD", "setup check", "retry"]:
        if expected not in body_out.output:
            raise SystemExit(f"read_email_body IMAP error missed recovery step {expected!r}: {body_out.output}")
    if body_out.metadata.get("exception_type") != "RuntimeError" or body_out.metadata.get("raw_limit") != "bad":
        raise SystemExit(f"read_email_body IMAP error should preserve bounded metadata: {body_out.metadata}")
    if not body_out.metadata.get("reads_private_data") or body_out.metadata.get("executes_side_effect"):
        raise SystemExit(f"read_email_body IMAP error metadata should preserve risk boundary: {body_out.metadata}")
    body_handoff = assert_email_body_handoff(body_out.metadata, "read_email_body IMAP failure", status="failed")
    if body_handoff.get("reason") != "body_error" or body_handoff.get("retry_safe") is not True or body_handoff.get("external_attempted") is not True:
        raise SystemExit(f"read_email_body IMAP failure handoff should preserve retryable external attempt: {body_handoff}")


def test_imap_rejected_statuses_never_become_empty_success() -> None:
    credentials = {"GMAIL_ADDRESS": "me@example.com", "GMAIL_APP_PASSWORD": "x" * 16}
    cases = [
        ("read_emails", {}, assert_email_read_handoff, "read_error"),
        ("search_emails", {"query": "invoice"}, assert_email_search_handoff, "search_error"),
        ("read_email_body", {"query": "invoice"}, assert_email_body_handoff, "body_error"),
    ]

    def fake_imap(failure_stage: str, failure_status: object):
        calls = {"select": 0, "search": 0, "fetch": 0}

        class StatusIMAP:
            def __init__(self, *args, **kwargs):
                pass

            def login(self, *args):
                if failure_stage == "login":
                    return (failure_status, [b"hostile login response"])
                return ("OK", [])

            def select(self, *args, **kwargs):
                calls["select"] += 1
                if failure_stage == "select":
                    return (failure_status, [b"hostile /\x55sers/private-select"])
                return ("OK", [b"1"])

            def search(self, *args):
                calls["search"] += 1
                if failure_stage == "search":
                    if failure_status == "MALFORMED_RESPONSE":
                        return None
                    return (failure_status, [b"hostile /private/search-result"])
                return ("OK", [b"1"])

            def fetch(self, *args):
                calls["fetch"] += 1
                if failure_stage == "fetch":
                    if failure_status == "EMPTY_OK":
                        return ("OK", [(b"1", b"")])
                    return (failure_status, [(b"1", b"From: secret /tmp/fetch-payload")])
                raw = b"From: Sender <sender@example.com>\r\nSubject: Invoice\r\n\r\nBody"
                return ("OK", [(b"1", raw)])

            def logout(self):
                return ("BYE", [])

        return StatusIMAP, calls

    for tool_name, args, assert_handoff, reason in cases:
        for stage, status, expected_status in (
            ("login", "NO", "NO"),
            ("select", b"NO", "NO"),
            ("search", "BAD", "BAD"),
            ("search", "MALFORMED_RESPONSE", "MALFORMED"),
            ("fetch", "NO", "NO"),
            ("fetch", "EMPTY_OK", "MALFORMED"),
        ):
            imap_class, calls = fake_imap(stage, status)
            with mock.patch.dict(os.environ, credentials, clear=False), \
                 mock.patch.object(ec.imaplib, "IMAP4_SSL", imap_class):
                out = _tools()[tool_name].handler(args)
            if out.ok or out.metadata.get("imap_operation") != stage or out.metadata.get("imap_status") != expected_status:
                raise SystemExit(
                    f"{tool_name} should fail closed on IMAP {stage} {status!r}: "
                    f"{out.output} {out.metadata}"
                )
            if "No matching emails" in out.output or "No emails" in out.output or "hostile" in str(out.metadata):
                raise SystemExit(f"{tool_name} converted protocol rejection into empty/leaky success: {out.output} {out.metadata}")
            if any(fragment in out.output or fragment in str(out.metadata) for fragment in LOCAL_PATH_FRAGMENTS):
                raise SystemExit(f"{tool_name} leaked rejected IMAP payload: {out.output} {out.metadata}")
            handoff = assert_handoff(out.metadata, f"{tool_name} {stage} {expected_status}", status="failed")
            if handoff.get("reason") != reason or handoff.get("retry_safe") is not True:
                raise SystemExit(f"{tool_name} rejected-status handoff was not retryable failure: {handoff}")
            if stage == "select" and (calls["search"] or calls["fetch"]):
                raise SystemExit(f"{tool_name} continued after rejected select: {calls}")
            if stage == "search" and calls["fetch"]:
                raise SystemExit(f"{tool_name} continued after rejected search: {calls}")


def test_read_email_body_mocked_plain_and_html() -> None:
    class FakeIMAP:
        def __init__(self, *a, **k):
            pass

        def login(self, *a):
            return ("OK", [])

        def select(self, *a, **k):
            return ("OK", [])

        def search(self, charset, criterion):
            return ("OK", [b"9"])

        def fetch(self, mid, spec):
            raw = (
                b"From: Alex </\x55sers/example/Desktop/Claude code/email-body-from.txt>\r\n"
                b"Subject: Project update near /private/tmp/email-body-subject and /var/folders/zc/email-body-subject and /tmp/email-body-subject\r\n"
                b"Date: Mon, 15 Jun 2026 09:00:00 +0000\r\n"
                b"Content-Type: text/html; charset=utf-8\r\n"
                b"\r\n"
                b"<html><body><p>Hello <b>the operator</b>,</p><p>The project shipped from /\x55sers/example/Desktop/Claude code/body.txt and /var/folders/zc/body.txt and /tmp/body.txt.</p></body></html>"
            )
            return ("OK", [(b"1", raw)])

        def logout(self):
            return ("BYE", [])

    tools = _tools()
    if tools["read_email_body"].risk != RiskLevel.PERSONAL_DATA:
        raise SystemExit("read_email_body must be PERSONAL_DATA gated")
    with mock.patch.dict(os.environ, {"GMAIL_ADDRESS": "me@example.com", "GMAIL_APP_PASSWORD": "x" * 16}, clear=False), \
         mock.patch.object(ec.imaplib, "IMAP4_SSL", FakeIMAP):
        out = tools["read_email_body"].handler({"sender": "alex@example.com", "limit": "bad"})
    if not out.ok or "Project update" not in out.output or "Hello the operator" not in out.output or "<b>" in out.output:
        raise SystemExit(f"read_email_body mocked output wrong: {out.output}")
    if any(fragment in out.output for fragment in ["/\x55sers/", "/private/", "/var/folders/", "/tmp/"]) or "<local-path>" not in out.output:
        raise SystemExit(f"read_email_body should redact path-shaped headers/body: {out.output}")
    if not out.metadata.get("reads_private_data") or out.metadata.get("executes_side_effect"):
        raise SystemExit(f"read_email_body metadata wrong: {out.metadata}")
    if out.metadata.get("limit") != 10 or out.metadata.get("raw_limit") != "bad":
        raise SystemExit(f"read_email_body should preserve sanitized raw limit metadata: {out.metadata}")
    handoff = assert_email_body_handoff(out.metadata, "read_email_body mocked", status="ok")
    if handoff.get("body_chars") != out.metadata.get("body_chars") or handoff.get("count") != 1:
        raise SystemExit(f"read_email_body handoff should preserve body length and match count: {handoff}")
    if "Project update" not in str(handoff.get("message")) or "Alex" not in str(handoff.get("message")):
        raise SystemExit(f"read_email_body handoff should preserve sanitized message preview: {handoff}")
    if "Hello the operator" in str(handoff):
        raise SystemExit(f"read_email_body handoff should not carry body text: {handoff}")
    with mock.patch.dict(os.environ, {"GMAIL_ADDRESS": "me@example.com", "GMAIL_APP_PASSWORD": "x" * 16}, clear=False), \
         mock.patch.object(ec.imaplib, "IMAP4_SSL", FakeIMAP):
        bool_out = tools["read_email_body"].handler({"sender": "alex@example.com", "limit": False, "index": True})
    if not bool_out.ok or bool_out.metadata.get("limit") != 10 or bool_out.metadata.get("raw_limit") != "False":
        raise SystemExit(f"read_email_body should treat boolean limits as malformed defaults: {bool_out.metadata}")
    if bool_out.metadata.get("selected_index") != 1:
        raise SystemExit(f"read_email_body should treat boolean index as the default first result: {bool_out.metadata}")
    assert_email_body_handoff(bool_out.metadata, "read_email_body boolean limit/index", status="ok")
    with mock.patch.dict(os.environ, {"GMAIL_ADDRESS": "me@example.com", "GMAIL_APP_PASSWORD": "x" * 16}, clear=False), \
         mock.patch.object(ec.imaplib, "IMAP4_SSL", FakeIMAP):
        path_out = tools["read_email_body"].handler({"sender": "alex@example.com", "limit": "/tmp/email-body-limit"})
    if not path_out.ok or path_out.metadata.get("limit") != 10 or path_out.metadata.get("raw_limit") != "<local-path>":
        raise SystemExit(f"read_email_body leaked local path in raw limit metadata: {path_out.metadata}")
    assert_email_body_handoff(path_out.metadata, "read_email_body path limit", status="ok")
    with mock.patch.dict(os.environ, {"GMAIL_ADDRESS": "me@example.com", "GMAIL_APP_PASSWORD": "x" * 16}, clear=False), \
         mock.patch.object(ec.imaplib, "IMAP4_SSL", FakeIMAP):
        bad_query = tools["read_email_body"].handler({"sender": "/var/folders/zc/email-body-query"})
    if bad_query.ok or bad_query.metadata.get("reason") != "invalid_search_term" or bad_query.metadata.get("criterion") != "<local-path>":
        raise SystemExit(f"read_email_body should reject path-shaped criteria locally: {bad_query.output} {bad_query.metadata}")
    if any(fragment in bad_query.output for fragment in ["/\x55sers/", "/private/", "/var/folders/", "/tmp/"]) or any(fragment in str(bad_query.metadata) for fragment in ["/\x55sers/", "/private/", "/var/folders/", "/tmp/"]):
        raise SystemExit(f"read_email_body path-shaped rejection leaked local paths: {bad_query.output} {bad_query.metadata}")
    bad_handoff = assert_email_body_handoff(bad_query.metadata, "read_email_body path query", status="refused")
    if bad_handoff.get("reason") != "invalid_search_term" or bad_handoff.get("external_attempted"):
        raise SystemExit(f"read_email_body path query handoff should be inert refusal: {bad_handoff}")


def test_body_and_search_require_credentials() -> None:
    tools = _tools()
    with mock.patch.dict(os.environ, {"GMAIL_ADDRESS": "", "GMAIL_APP_PASSWORD": ""}, clear=False):
        for name in ["search_emails", "read_email_body"]:
            out = tools[name].handler({"query": "invoice", "limit": "l" * 120})
            if out.ok or "GMAIL" not in out.output:
                raise SystemExit(f"{name} should require credentials: {out.output}")
            if out.metadata.get("raw_limit") != ("l" * 80):
                raise SystemExit(f"{name} should bound raw credential limit metadata: {out.metadata}")
            if name == "read_email_body":
                handoff = assert_email_body_handoff(out.metadata, "read_email_body missing credentials", status="unavailable")
                if handoff.get("reason") != "missing_credentials" or handoff.get("external_attempted"):
                    raise SystemExit(f"read_email_body missing credentials should be inert unavailable state: {handoff}")


def test_planner_routes_email_search_and_body_read() -> None:
    planner = RuleBasedPlanner()
    search_plan = planner.plan("search emails about invoice")
    if search_plan.actions[0].tool_name != "search_emails" or search_plan.actions[0].args.get("query") != "invoice":
        raise SystemExit(f"planner missed email search route: {search_plan}")
    sender_plan = planner.plan("email from Sam")
    if sender_plan.actions[0].tool_name != "search_emails" or sender_plan.actions[0].args.get("sender") != "Sam":
        raise SystemExit(f"planner missed bare sender email search route: {sender_plan}")
    mail_sender_plan = planner.plan("mail from Sam")
    if mail_sender_plan.actions[0].tool_name != "search_emails" or mail_sender_plan.actions[0].args.get("sender") != "Sam":
        raise SystemExit(f"planner missed bare sender mail search route: {mail_sender_plan}")
    find_mail_sender_plan = planner.plan("find mail from Sam")
    if find_mail_sender_plan.actions[0].tool_name != "search_emails" or find_mail_sender_plan.actions[0].args.get("sender") != "Sam":
        raise SystemExit(f"planner missed find-mail sender route: {find_mail_sender_plan}")
    sender_today_plan = planner.plan("emails from Sam today")
    if sender_today_plan.actions[0].tool_name != "search_emails" or sender_today_plan.actions[0].args.get("sender") != "Sam":
        raise SystemExit(f"planner should strip date hints from email sender route: {sender_today_plan}")
    sent_by_plan = planner.plan("what emails did Sam send me")
    if sent_by_plan.actions[0].tool_name != "search_emails" or sent_by_plan.actions[0].args.get("sender") != "Sam":
        raise SystemExit(f"planner missed what-emails-did-sender-send route: {sent_by_plan}")
    sent_by_without_me_plan = planner.plan("what emails did Sam send")
    if sent_by_without_me_plan.actions[0].tool_name != "search_emails" or sent_by_without_me_plan.actions[0].args.get("sender") != "Sam":
        raise SystemExit(f"planner missed what-emails-did-sender-send route without me: {sent_by_without_me_plan}")
    emails_sender_plan = planner.plan("emails from Sam about invoice")
    if emails_sender_plan.actions[0].tool_name != "search_emails" or emails_sender_plan.actions[0].args.get("sender") != "Sam" or emails_sender_plan.actions[0].args.get("query") != "invoice":
        raise SystemExit(f"planner missed sender+query email search route: {emails_sender_plan}")
    mail_query_plan = planner.plan("search mail about invoice")
    if mail_query_plan.actions[0].tool_name != "search_emails" or mail_query_plan.actions[0].args.get("query") != "invoice":
        raise SystemExit(f"planner missed mail query search route: {mail_query_plan}")
    bare_mail_query_plan = planner.plan("mail about invoice")
    if bare_mail_query_plan.actions[0].tool_name != "search_emails" or bare_mail_query_plan.actions[0].args.get("query") != "invoice":
        raise SystemExit(f"planner missed bare mail query route: {bare_mail_query_plan}")
    subject_plan = planner.plan("email subject receipt")
    if subject_plan.actions[0].tool_name != "search_emails" or subject_plan.actions[0].args.get("subject") != "receipt":
        raise SystemExit(f"planner missed bare subject email search route: {subject_plan}")

    send_plan = planner.plan("send email to alex@example.com saying Jarvis email delivery proof")
    if (
        len(send_plan.actions) != 1
        or send_plan.actions[0].tool_name != "send_email"
        or send_plan.actions[0].args
        != {"to": "alex@example.com", "body": "Jarvis email delivery proof"}
    ):
        raise SystemExit(f"planner missed natural email send route: {send_plan}")
    if _tools()["send_email"].risk != RiskLevel.HIGH_RISK:
        raise SystemExit("natural email send route must retain the HIGH_RISK approval gate")
    named_send_plan = planner.plan("send an email to Fixture Example saying hello")
    if (
        len(named_send_plan.actions) != 1
        or named_send_plan.actions[0].tool_name != "send_email"
        or named_send_plan.actions[0].args != {"to": "Fixture Example", "body": "hello"}
    ):
        raise SystemExit(f"planner missed contact-name email send route: {named_send_plan}")
    mail_subject_plan = planner.plan("mail subject receipt")
    if mail_subject_plan.actions[0].tool_name != "search_emails" or mail_subject_plan.actions[0].args.get("subject") != "receipt":
        raise SystemExit(f"planner missed bare subject mail search route: {mail_subject_plan}")
    body_plan = planner.plan("read the email from Alex about project")
    action = body_plan.actions[0]
    if action.tool_name != "read_email_body":
        raise SystemExit(f"planner missed email body route: {body_plan}")
    if action.args.get("sender") != "Alex" or action.args.get("query") != "project":
        raise SystemExit(f"planner did not preserve body sender hint: {action.args}")
    mail_body_plan = planner.plan("read mail from Sam about invoice")
    action = mail_body_plan.actions[0]
    if action.tool_name != "read_email_body" or action.args.get("sender") != "Sam" or action.args.get("query") != "invoice":
        raise SystemExit(f"planner missed mail body sender+query route: {mail_body_plan}")
    open_email_body_plan = planner.plan("open email from Sam")
    action = open_email_body_plan.actions[0]
    if action.tool_name != "read_email_body" or action.args.get("sender") != "Sam":
        raise SystemExit(f"planner should route open-email sender to read_email_body, not open_application: {open_email_body_plan}")
    # Real gap found live 2026-07-10: the email-search trigger matched the
    # bare word "email"/"mail" appearing ANYWHERE within 30 characters of
    # "search"/"find" -- the same unanchored-substring-collision class as the
    # round-21 "meeting" bug and round-22 "google" bug -- so natural phrases
    # like "find a way to backup my email" (asking for advice, not a search)
    # misrouted to search_emails with empty args. Tightened the gate to
    # require "search"/"find" directly govern the email noun (with only a
    # small set of real modifier words in between), matching how "search my
    # emails" and "find emails from john" are actually phrased.
    for text in (
        "find a way to backup my email",
        "find out if john's email address changed",
        "search for a way to organize my email",
    ):
        non_email_plan = planner.plan(text)
        if [a.tool_name for a in non_email_plan.actions] == ["search_emails"]:
            raise SystemExit(f"non-email-search phrase should not misroute to search_emails: {text!r} -> {non_email_plan.actions}")
    for text in ("search my emails", "search for emails about invoice", "find recent emails"):
        genuine_email_plan = planner.plan(text)
        if [a.tool_name for a in genuine_email_plan.actions] != ["search_emails"]:
            raise SystemExit(f"genuine email-search phrase should still route to search_emails: {text!r} -> {genuine_email_plan.actions}")
    read_aliases = {
        "email please": {},
        "emails please": {},
        "mail please": {},
        "inbox please": {},
        "show email": {},
        "show inbox": {},
        "any new mail": {"unread": True},
        "do I have unread mail": {"unread": True},
        "latest mail": {},
        "read recent mail": {},
        "read latest mail": {},
    }
    for text, expected_args in read_aliases.items():
        read_plan = planner.plan(text)
        if [(a.tool_name, a.args) for a in read_plan.actions] != [("read_emails", expected_args)]:
            raise SystemExit(f"planner missed email read alias {text!r}: {read_plan.actions}")


def main() -> None:
    test_requires_credentials()
    test_path_shaped_credentials_are_rejected_locally()
    test_requires_recipient_and_body()
    test_send_redacts_path_shaped_recipient_metadata()
    test_successful_send_is_mocked()
    test_send_error_preserves_safe_metadata()
    test_send_auth_error_names_the_fix()
    test_read_auth_error_names_the_fix()
    test_generic_email_error_keeps_retry_message()
    test_sendmail_error_preserves_attempt_metadata()
    test_send_resolves_contact_names_before_smtp()
    test_send_ambiguous_contact_refuses_without_smtp()
    test_send_missing_contact_refuses_without_smtp()
    test_send_malformed_contact_resolution_defaults_closed()
    test_send_explicit_email_skips_contact_resolution()
    test_read_requires_credentials()
    test_read_emails_mocked()
    test_read_emails_empty_handoff()
    test_read_email_failures_are_clean()
    test_search_emails_mocked_and_personal_data_gated()
    test_unicode_email_search_uses_utf8_literal()
    test_unicode_email_search_preserves_structured_gmail_terms()
    test_unicode_email_body_search_uses_same_utf8_literal_path()
    test_unicode_email_search_records_safe_bad_stage()
    test_search_emails_empty_handoff()
    test_read_email_body_empty_handoff()
    test_search_and_body_failures_are_clean()
    test_imap_rejected_statuses_never_become_empty_success()
    test_read_email_body_mocked_plain_and_html()
    test_body_and_search_require_credentials()
    test_planner_routes_email_search_and_body_read()
    print("Email connector smoke passed")


if __name__ == "__main__":
    main()
