"""Focused offline proof for canonical Gmail failure guidance and send truth."""

from __future__ import annotations

import os
from unittest import mock

from jarvis_v2.config import load_config
from jarvis_v2.tools import email_connector as ec


LOCAL_PATH_FRAGMENTS = ("/\x55sers/", "/private/", "/var/folders/", "/tmp/")
SAFE_CREDENTIALS = {
    "GMAIL_ADDRESS": "smoke@example.com",
    "GMAIL_APP_PASSWORD": "x" * 16,
}


def _tools():
    return {tool.name: tool for tool in ec.make_email_tools(load_config())}


def _assert_private_safe(result, label: str) -> None:
    rendered = f"{result.output}\n{result.metadata}"
    if any(fragment in rendered for fragment in LOCAL_PATH_FRAGMENTS):
        raise SystemExit(f"{label} leaked a local path: {rendered}")
    if "ghp" + "_private-email-guidance-sentinel" in rendered:
        raise SystemExit(f"{label} leaked secret-shaped input: {rendered}")


def _assert_guidance(
    result,
    label: str,
    *,
    action: str,
    commands: list[str],
    retry_safe: bool | None,
) -> None:
    if result.ok or action not in result.output:
        raise SystemExit(f"{label} missed its user-visible recovery action: {result}")
    expected_guidance = {"version": 1, "action": action, "commands": commands}
    if result.metadata.get("recovery_guidance") != expected_guidance:
        raise SystemExit(f"{label} canonical recovery declaration drifted: {result.metadata}")
    expected = {
        "outcome_known": True,
        "outcome_unknown": False,
        "execution_outcome_unknown": False,
        "side_effect_possible": False,
        "automatic_retry_allowed": False,
        "authorizes_retry": False,
        "authorizes_execution": False,
        "authorizes_completion_claim": False,
        "approval_granted": False,
    }
    if retry_safe is not None:
        expected["retry_safe"] = retry_safe
    for key, value in expected.items():
        if result.metadata.get(key) is not value:
            raise SystemExit(f"{label} field {key} should be {value}: {result.metadata}")
    if commands:
        if result.metadata.get("next_command") != commands[0]:
            raise SystemExit(f"{label} next command drifted: {result.metadata}")
        if result.metadata.get("recovery_commands") != commands:
            raise SystemExit(f"{label} recovery command list drifted: {result.metadata}")
    _assert_private_safe(result, label)


def test_send_preflight_failures_are_known_not_sent() -> None:
    tools = _tools()
    with mock.patch.dict(
        os.environ,
        {"GMAIL_ADDRESS": "", "GMAIL_APP_PASSWORD": "ghp" + "_private-email-guidance-sentinel"},
        clear=False,
    ), mock.patch.object(
        ec.smtplib,
        "SMTP_SSL",
        side_effect=AssertionError("credential failure must not start SMTP"),
    ):
        credentials = tools["send_email"].handler(
            {"to": "alex@example.com", "subject": "Proof", "body": "hello"}
        )
    _assert_guidance(
        credentials,
        "send credential failure",
        action=ec.KNOWN_NOT_SENT_SEND_RECOVERY_ACTION,
        commands=["setup check"],
        retry_safe=None,
    )
    if credentials.metadata.get("external_attempted") is not False:
        raise SystemExit(f"credential failure claimed external work: {credentials.metadata}")

    with mock.patch.dict(os.environ, SAFE_CREDENTIALS, clear=False), mock.patch.object(
        ec.smtplib,
        "SMTP_SSL",
        side_effect=AssertionError("validation failure must not start SMTP"),
    ), mock.patch.object(
        ec.contacts_connector,
        "resolve_contact",
        return_value=[],
    ):
        cases = (
            tools["send_email"].handler({"subject": "Proof", "body": "hello"}),
            tools["send_email"].handler({"to": "alex@example.com", "subject": "Proof"}),
            tools["send_email"].handler({"to": "Fixture", "subject": "Proof", "body": "hello"}),
        )
    for label, result in zip(
        ("missing recipient", "missing body", "unresolved contact"),
        cases,
    ):
        _assert_guidance(
            result,
            label,
            action=ec.KNOWN_NOT_SENT_SEND_RECOVERY_ACTION,
            commands=["setup check"],
            retry_safe=None,
        )
        if result.metadata.get("external_attempted", False) or result.metadata.get(
            "send_attempted", False
        ):
            raise SystemExit(f"{label} claimed an SMTP/send attempt: {result.metadata}")


def test_read_input_failures_are_retryable_and_inert() -> None:
    tools = _tools()
    with mock.patch.dict(os.environ, SAFE_CREDENTIALS, clear=False), mock.patch.object(
        ec.imaplib,
        "IMAP4_SSL",
        side_effect=AssertionError("invalid read input must not start IMAP"),
    ):
        cases = (
            tools["search_emails"].handler({"query": "/\x55sers/example/private/mail"}),
            tools["read_email_body"].handler({"query": "invoice", "limit": 2, "index": 3}),
            tools["read_email_body"].handler({"query": "invoice\x00private"}),
        )
    for label, result in zip(
        ("search path", "body index exceeds limit", "body control character"),
        cases,
    ):
        _assert_guidance(
            result,
            label,
            action=ec.PERSONAL_READ_RECOVERY_ACTION,
            commands=["setup check"],
            retry_safe=True,
        )
        if result.metadata.get("external_attempted", False):
            raise SystemExit(f"{label} claimed IMAP work: {result.metadata}")


def test_missing_selected_message_keeps_read_only_truth() -> None:
    class OneMessageIMAP:
        def __init__(self, host: str, port: int):
            if (host, port) != ("imap.gmail.com", 993):
                raise AssertionError(f"unexpected IMAP endpoint: {host}:{port}")

        def login(self, _address: str, _password: str):
            return "OK", []

        def select(self, mailbox: str, *, readonly: bool = False):
            if mailbox != "INBOX" or readonly is not True:
                raise AssertionError("email body guidance proof must open INBOX read-only")
            return "OK", []

        def search(self, charset, criterion):
            if charset is not None or criterion != 'TEXT "invoice"':
                raise AssertionError(f"unexpected search: {charset!r} {criterion!r}")
            return "OK", [b"17"]

        def fetch(self, *_args, **_kwargs):
            raise AssertionError("unavailable selection must not fetch a different message")

        def logout(self):
            return "BYE", []

    with mock.patch.dict(os.environ, SAFE_CREDENTIALS, clear=False), mock.patch.object(
        ec.imaplib,
        "IMAP4_SSL",
        OneMessageIMAP,
    ):
        result = _tools()["read_email_body"].handler(
            {"query": "invoice", "limit": 3, "index": 2}
        )
    _assert_guidance(
        result,
        "missing selected message",
        action=ec.PERSONAL_READ_RECOVERY_ACTION,
        commands=["setup check"],
        retry_safe=True,
    )
    if result.metadata.get("external_attempted") is not True:
        raise SystemExit(f"missing selection hid its read-only IMAP attempt: {result.metadata}")
    boundaries = result.metadata.get("email_body_boundaries", {})
    if boundaries.get("calls_external_service") is not True:
        raise SystemExit(f"missing selection handoff hid the IMAP attempt: {result.metadata}")
    if boundaries.get("executes_side_effect") or boundaries.get("external_side_effect"):
        raise SystemExit(f"missing selection claimed a side effect: {result.metadata}")


def test_post_send_transport_failure_remains_outcome_unknown() -> None:
    class UncertainSMTP:
        def __init__(self, *_args, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def login(self, _address: str, _password: str):
            return None

        def sendmail(self, *_args, **_kwargs):
            raise ec.smtplib.SMTPServerDisconnected("private transport detail")

    with mock.patch.dict(os.environ, SAFE_CREDENTIALS, clear=False), mock.patch.object(
        ec.smtplib,
        "SMTP_SSL",
        UncertainSMTP,
    ):
        result = _tools()["send_email"].handler(
            {"to": "alex@example.com", "subject": "Proof", "body": "hello"}
        )
    if result.ok or "do not retry blindly" not in result.output or "Check Gmail Sent" not in result.output:
        raise SystemExit(f"post-send failure did not require state verification: {result}")
    expected = {
        "outcome_known": False,
        "outcome_unknown": True,
        "execution_outcome_unknown": True,
        "side_effect_possible": True,
        "retry_safe": False,
        "automatic_retry_allowed": False,
        "authorizes_retry": False,
        "send_attempted": True,
        "external_attempted": True,
    }
    for key, value in expected.items():
        if result.metadata.get(key) is not value:
            raise SystemExit(f"outcome-unknown send field {key} drifted: {result.metadata}")
    if result.metadata.get("recovery_guidance") != {
        "version": 1,
        "action": result.output,
        "commands": ["setup check"],
    }:
        raise SystemExit(f"outcome-unknown send guidance drifted: {result.metadata}")
    if "private transport detail" in result.output or "private transport detail" in str(
        result.metadata
    ):
        raise SystemExit("outcome-unknown send leaked the transport exception")


def main() -> None:
    test_send_preflight_failures_are_known_not_sent()
    test_read_input_failures_are_retryable_and_inert()
    test_missing_selected_message_keeps_read_only_truth()
    test_post_send_transport_failure_remains_outcome_unknown()
    print("Email failure guidance smoke test passed.")


if __name__ == "__main__":
    main()
