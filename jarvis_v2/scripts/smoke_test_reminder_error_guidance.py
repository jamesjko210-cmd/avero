"""Offline smoke for canonical reminder refusal and recovery guidance."""

from __future__ import annotations

import hashlib
import hmac
import os
import time
from unittest.mock import patch

from jarvis_v2.agent.types import ToolResult
from jarvis_v2.automations.reminders import ReminderWriteResult
from jarvis_v2.config import load_config
from jarvis_v2.tools import reminder_tools


PRIVATE_MARKERS = (
    "/\x55sers/owner/private-reminder.txt",
    "/private/var/folders/owner-reminder-state",
)


def _assert_guidance(result: ToolResult, *, label: str) -> None:
    if result.ok is not False:
        raise SystemExit(f"{label} unexpectedly succeeded")
    declaration = result.metadata.get("recovery_guidance")
    if not isinstance(declaration, dict) or declaration.get("version") != 1:
        raise SystemExit(f"{label} missed canonical recovery guidance: {result.metadata}")
    action = declaration.get("action")
    if not isinstance(action, str) or action not in result.output:
        raise SystemExit(f"{label} recovery action is not user-visible: {result}")
    for command in declaration.get("commands") or []:
        if command not in result.output:
            raise SystemExit(f"{label} recovery command is not user-visible: {result}")
    expected_truth = {
        "outcome_known": True,
        "outcome_unknown": False,
        "execution_outcome_unknown": False,
        "side_effect_possible": False,
        "retry_safe": True,
        "automatic_retry_allowed": False,
        "authorizes_retry": False,
        "authorizes_execution": False,
        "authorizes_completion_claim": False,
        "approval_granted": False,
    }
    for key, value in expected_truth.items():
        if result.metadata.get(key) is not value:
            raise SystemExit(f"{label} recovery truth drifted for {key}: {result.metadata}")
    public = f"{result.output}\n{declaration}\n{result.metadata}"
    if any(marker in public for marker in PRIVATE_MARKERS):
        raise SystemExit(f"{label} leaked private detail: {public}")


def main() -> None:
    tools = {tool.name: tool for tool in reminder_tools.make_reminder_tools(load_config())}
    set_tool = tools["set_reminder"]
    location_tool = tools["location_reminder_draft"]
    cancel_tool = tools["cancel_reminders"]
    resolver = set_tool.approval_argument_resolver
    if resolver is None:
        raise SystemExit("set_reminder approval resolver is missing")

    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("JARVIS_OWNER_TELEGRAM", None)
        os.environ.pop("TELEGRAM_BOT_TOKEN", None)
        for label, args in (
            ("missing text", {}),
            ("recurring", {"text": "remind me every day to stretch"}),
            ("unparsed time", {"text": "remind me eventually to stretch"}),
            ("local-path text", {"text": f"remind me in 20 minutes to read {PRIVATE_MARKERS[0]}"}),
            ("missing owner", {"text": "remind me in 20 minutes to stretch"}),
        ):
            result = resolver(args)
            if not isinstance(result, ToolResult):
                raise SystemExit(f"set reminder {label} unexpectedly resolved approval args")
            _assert_guidance(result, label=f"set reminder {label}")

        os.environ["JARVIS_OWNER_TELEGRAM"] = "offline-owner"
        missing_token = resolver({"text": "remind me in 20 minutes to stretch"})
        if not isinstance(missing_token, ToolResult):
            raise SystemExit("set reminder missing token unexpectedly resolved approval args")
        _assert_guidance(missing_token, label="set reminder missing token")

        os.environ["TELEGRAM_BOT_TOKEN"] = "offline-token"
        due_epoch = time.time() + 3600
        stale = set_tool.handler(
            {
                "due_epoch": due_epoch,
                "message": "stretch",
                "owner_fingerprint": "not-the-owner-fingerprint",
            }
        )
        _assert_guidance(stale, label="set reminder stale binding")

        fingerprint = hmac.new(
            b"offline-token",
            b"offline-owner",
            hashlib.sha256,
        ).hexdigest()
        stale_due = set_tool.handler(
            {
                "due_epoch": time.time() - 60,
                "message": "stretch",
                "owner_fingerprint": fingerprint,
            }
        )
        _assert_guidance(stale_due, label="set reminder stale due")

        with patch.object(
            reminder_tools,
            "add_reminder_result",
            return_value=ReminderWriteResult("failed", reason="write_error"),
        ):
            failed_write = set_tool.handler(
                {
                    "due_epoch": due_epoch,
                    "message": "stretch",
                    "owner_fingerprint": fingerprint,
                }
            )
        _assert_guidance(failed_write, label="set reminder storage failure")

        os.environ.pop("JARVIS_OWNER_TELEGRAM", None)
        _assert_guidance(cancel_tool.handler({}), label="cancel reminder missing owner")

        os.environ["JARVIS_OWNER_TELEGRAM"] = "offline-owner"
        with patch.object(
            reminder_tools._rem,
            "cancel_pending_item",
            return_value=("storage_error", [], False),
        ):
            _assert_guidance(
                cancel_tool.handler({"reminder_id": "deadbeef"}),
                label="cancel reminder storage failure",
            )

        for status in ("invalid_selector", "not_found", "ambiguous", "in_flight"):
            with patch.object(
                reminder_tools._rem,
                "cancel_pending_item",
                return_value=(status, [], True),
            ):
                _assert_guidance(
                    cancel_tool.handler({"reminder_id": "deadbeef"}),
                    label=f"cancel reminder {status}",
                )

        with patch.object(
            reminder_tools._rem,
            "cancel_pending_items_status",
            return_value=([], True, 1, "in_flight"),
        ):
            _assert_guidance(cancel_tool.handler({}), label="cancel all in flight")

    for label, args in (
        ("missing text", {}),
        ("missing message", {"text": "when I get home"}),
        ("local path", {"text": f"when I get home to read {PRIVATE_MARKERS[0]}"}),
        ("unsupported", {"text": "when the moon rises remind me to stretch"}),
    ):
        _assert_guidance(location_tool.handler(args), label=f"location reminder {label}")

    print("Reminder error-guidance smoke passed: 20 pre-change failure paths")


if __name__ == "__main__":
    main()
