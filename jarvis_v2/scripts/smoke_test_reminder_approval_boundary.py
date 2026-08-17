from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from jarvis_v2.agent.types import PlannedAction, RiskLevel
from jarvis_v2.automations import reminders
from jarvis_v2.scripts.test_runtime import (
    approve_pending_runtime_approval,
    make_temp_runtime,
)


FAKE_OWNER_CHAT_ID = "reminder-approval-smoke-owner"
REQUEST = "remind me tomorrow at 9am to stretch for approval boundary smoke"


def _reminder_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise SystemExit(f"Reminder store did not contain a row list: {rows!r}")
    return rows


def _held_approval_id(result, *, reused: bool) -> int:
    if len(result.plan.actions) != 1 or result.plan.actions[0].tool_name != "set_reminder":
        raise SystemExit(f"Reminder request did not plan exactly one set_reminder action: {result.plan}")
    if len(result.tool_results) != 1:
        raise SystemExit(f"Reminder hold returned the wrong result count: {result.tool_results}")

    held = result.tool_results[0]
    approval_id = held.metadata.get("approval_id")
    if (
        result.verified
        or held.ok
        or held.metadata.get("failure_kind") != "approval_required"
        or held.metadata.get("requires_confirmation") is not True
        or held.metadata.get("executed_handler") is not False
        or held.metadata.get("reused_pending_approval") is not reused
        or type(approval_id) is not int
    ):
        raise SystemExit(f"set_reminder crossed or malformed its approval hold: {result}")
    return approval_id


def assert_reminder_approval_boundary() -> None:
    with TemporaryDirectory(prefix="jarvis-reminder-approval-boundary-") as temp:
        root = Path(temp)
        reminders_file = root / "reminders.json"

        with (
            patch.dict(
                os.environ,
                {
                    "JARVIS_OWNER_TELEGRAM": FAKE_OWNER_CHAT_ID,
                    "TELEGRAM_BOT_TOKEN": "fake-token-for-reminder-approval-smoke",
                },
            ),
            patch.object(reminders, "REMINDERS_FILE", reminders_file),
        ):
            runtime = make_temp_runtime(root)
            set_reminder = runtime.registry.get("set_reminder")
            if set_reminder.risk is not RiskLevel.EXTERNAL_SIDE_EFFECT:
                raise SystemExit(
                    "set_reminder must be registered as EXTERNAL_SIDE_EFFECT so the runtime approval policy holds it; "
                    f"found {set_reminder.risk.name}."
                )
            resolver = set_reminder.approval_argument_resolver
            if resolver is None:
                raise SystemExit("set_reminder missed its pre-approval resolver")
            minute_due = datetime(2030, 1, 1, 12, 1, 59, 900000).astimezone()
            with patch("jarvis_v2.tools.reminder_tools.parse_when", return_value=(minute_due, "minute test")):
                minute_binding = resolver({"text": "remind me in 1 minute to test"})
            expected_minute = minute_due.replace(second=0, microsecond=0).timestamp() + 60
            if getattr(minute_binding, "args", {}).get("due_epoch") != expected_minute:
                raise SystemExit(f"Minute reminder binding rounded early: {minute_binding}")

            second_due = datetime(2030, 1, 1, 12, 1, 9, 900000).astimezone()
            with patch("jarvis_v2.tools.reminder_tools.parse_when", return_value=(second_due, "second test")):
                second_binding = resolver({"text": "remind me in 1 second to test"})
            expected_second = second_due.replace(microsecond=0).timestamp() + 1
            if getattr(second_binding, "args", {}).get("due_epoch") != expected_second:
                raise SystemExit(f"Second reminder binding rounded early: {second_binding}")

            invalid_args = (
                {},
                {"request": REQUEST},
                {"text": 20},
                {"text": REQUEST, "extra": True},
            )
            for args in invalid_args:
                invalid = runtime.executor.execute(PlannedAction("set_reminder", args))
                if (
                    invalid.ok
                    or invalid.metadata.get("failure_kind") != "tool_arguments_invalid"
                    or invalid.metadata.get("executed_handler") is not False
                    or _reminder_rows(reminders_file)
                ):
                    raise SystemExit(f"Malformed reminder args crossed the strict pre-approval gate: {args!r} -> {invalid}")

            for command in (
                "remind me every day at 9am to stretch",
                "remind me in 20 minutes to open /\x55sers/the operator/private.txt",
                "remind me sometime to stretch",
            ):
                refused = runtime.handle(command)
                if (
                    any(result.metadata.get("requires_confirmation") for result in refused.tool_results)
                    or runtime.store.list_pending_approvals(limit=100)
                    or _reminder_rows(reminders_file)
                ):
                    raise SystemExit(f"Invalid reminder intent reached approval persistence or storage: {command!r} -> {refused}")

            with patch.dict(os.environ, {"TELEGRAM_BOT_TOKEN": ""}):
                missing_token = runtime.handle("remind me in 21 minutes to stretch")
            if (
                any(result.metadata.get("requires_confirmation") for result in missing_token.tool_results)
                or runtime.store.list_pending_approvals(limit=100)
                or _reminder_rows(reminders_file)
            ):
                raise SystemExit(f"Missing Telegram token reached approval persistence or storage: {missing_token}")

            first_hold = runtime.handle(REQUEST)
            approval_id = _held_approval_id(first_hold, reused=False)
            approval_row = runtime.store.get_pending_approval(approval_id)
            if approval_row is None:
                raise SystemExit("set_reminder approval row was not persisted")
            bound_args = json.loads(str(approval_row["planned_args"] or "{}"))
            if (
                set(bound_args) != {"due_epoch", "message", "owner_fingerprint"}
                or bound_args.get("message") != "stretch for approval boundary smoke"
                or not isinstance(bound_args.get("due_epoch"), (int, float))
                or not isinstance(bound_args.get("owner_fingerprint"), str)
                or FAKE_OWNER_CHAT_ID in str(bound_args)
            ):
                raise SystemExit(f"set_reminder approval did not bind a private exact target: {bound_args}")
            if _reminder_rows(reminders_file):
                raise SystemExit("Unapproved set_reminder created a reminder row.")

            duplicate_hold = runtime.handle(REQUEST)
            duplicate_id = _held_approval_id(duplicate_hold, reused=True)
            pending = runtime.store.list_pending_approvals(limit=100)
            if duplicate_id != approval_id or len(pending) != 1 or int(pending[0]["id"]) != approval_id:
                raise SystemExit(
                    "Identical pending set_reminder request did not reuse exactly one approval: "
                    f"first={approval_id}, duplicate={duplicate_id}, pending={[dict(row) for row in pending]}"
                )
            if _reminder_rows(reminders_file):
                raise SystemExit("Deduplicated unapproved set_reminder created a reminder row.")

            readiness = runtime.registry.get("approval_readiness_packet").handler(
                {"approval_id": approval_id}
            )
            packet = runtime.registry.get("approval_execution_packet").handler(
                {"approval_id": approval_id}
            )
            expected_due_display = datetime.fromtimestamp(
                float(bound_args["due_epoch"])
            ).astimezone().isoformat(timespec="seconds")
            if (
                not readiness.ok
                or not packet.ok
                or "owner_fingerprint: bound to configured Telegram owner" not in packet.output
                or f"due_epoch: {expected_due_display}" not in packet.output
                or bound_args["owner_fingerprint"] in packet.output
            ):
                raise SystemExit(f"Reminder approval packet did not render a safe exact target: {packet}")
            packet_display = packet.metadata.get("planned_args", {})
            handoff_display = packet.metadata.get("approval_execution_handoff", {}).get("planned_args", {})
            for label, display in (("packet", packet_display), ("handoff", handoff_display)):
                if (
                    display.get("owner_fingerprint") != "bound to configured Telegram owner"
                    or display.get("due_epoch") != expected_due_display
                    or bound_args["owner_fingerprint"] in json.dumps(display, sort_keys=True)
                ):
                    raise SystemExit(f"Reminder {label} metadata exposed an opaque binding: {display}")

            transition = approve_pending_runtime_approval(runtime, approval_id)
            transition_display = transition.metadata.get("exact_rerun_args_display", {})
            if (
                transition_display.get("owner_fingerprint") != "bound to configured Telegram owner"
                or transition_display.get("due_epoch") != expected_due_display
                or bound_args["owner_fingerprint"] in json.dumps(transition_display, sort_keys=True)
            ):
                raise SystemExit(f"Reminder approval transition exposed an opaque binding: {transition_display}")
            if _reminder_rows(reminders_file):
                raise SystemExit("Approval transition created a reminder before the exact approved rerun.")

            approved = runtime.handle(REQUEST, approved=True, approved_approval_id=approval_id)
            if (
                not approved.verified
                or len(approved.plan.actions) != 1
                or approved.plan.actions[0].tool_name != "set_reminder"
                or len(approved.tool_results) != 1
                or approved.tool_results[0].tool_name != "set_reminder"
                or approved.tool_results[0].ok is not True
                or approved.tool_results[0].metadata.get("handler_invoked") is not True
            ):
                raise SystemExit(f"Approved set_reminder did not execute exactly once: {approved}")

            rows = _reminder_rows(reminders_file)
            if (
                len(rows) != 1
                or rows[0].get("message") != "stretch for approval boundary smoke"
                or rows[0].get("chat_id") != FAKE_OWNER_CHAT_ID
                or rows[0].get("state") != "pending"
                or rows[0].get("attempt_count") != 0
                or abs(float(rows[0].get("due", 0)) - float(bound_args["due_epoch"])) > 0.001
            ):
                raise SystemExit(f"Approved set_reminder did not create exactly one expected row: {rows}")

            repeated_approval = runtime.registry.get("approve_pending_approval").handler(
                {"approval_id": approval_id}
            )
            if repeated_approval.ok or len(_reminder_rows(reminders_file)) != 1:
                raise SystemExit(
                    "Repeated approval was accepted or changed reminder rows: "
                    f"approval={repeated_approval}, rows={_reminder_rows(reminders_file)}"
                )

            replay = runtime.handle(REQUEST, approved=True, approved_approval_id=approval_id)
            if replay.verified or replay.plan.actions or replay.tool_results or len(_reminder_rows(reminders_file)) != 1:
                raise SystemExit(
                    "Consumed approval replay executed or changed reminder rows: "
                    f"replay={replay}, rows={_reminder_rows(reminders_file)}"
                )

            fresh_hold = runtime.handle(REQUEST)
            fresh_approval_id = _held_approval_id(fresh_hold, reused=False)
            if fresh_approval_id == approval_id or len(_reminder_rows(reminders_file)) != 1:
                raise SystemExit(
                    "A new reminder request reused consumed authority or changed reminder rows before approval: "
                    f"old={approval_id}, fresh={fresh_approval_id}, rows={_reminder_rows(reminders_file)}"
                )


def assert_owner_binding_cannot_redirect() -> None:
    with TemporaryDirectory(prefix="jarvis-reminder-owner-binding-") as temp:
        root = Path(temp)
        reminders_file = root / "reminders.json"
        with (
            patch.dict(
                os.environ,
                {
                    "JARVIS_OWNER_TELEGRAM": "owner-a",
                    "TELEGRAM_BOT_TOKEN": "fake-token-for-owner-binding-smoke",
                },
            ),
            patch.object(reminders, "REMINDERS_FILE", reminders_file),
        ):
            runtime = make_temp_runtime(root)
            held = runtime.handle(REQUEST)
            approval_id = _held_approval_id(held, reused=False)
            approve_pending_runtime_approval(runtime, approval_id)
            with patch.dict(os.environ, {"JARVIS_OWNER_TELEGRAM": "owner-b"}):
                redirected = runtime.handle(REQUEST, approved=True, approved_approval_id=approval_id)
            if (
                redirected.verified
                or len(redirected.tool_results) != 1
                or redirected.tool_results[0].metadata.get("reason") != "stale_reminder_binding"
                or _reminder_rows(reminders_file)
            ):
                raise SystemExit(f"Changed owner destination escaped the approval binding: {redirected}")


def assert_concurrent_pending_dedupe() -> None:
    with TemporaryDirectory(prefix="jarvis-reminder-concurrent-hold-") as temp:
        root = Path(temp)
        reminders_file = root / "reminders.json"
        with (
            patch.dict(
                os.environ,
                {
                    "JARVIS_OWNER_TELEGRAM": FAKE_OWNER_CHAT_ID,
                    "TELEGRAM_BOT_TOKEN": "fake-token-for-concurrent-reminder-smoke",
                },
            ),
            patch.object(reminders, "REMINDERS_FILE", reminders_file),
        ):
            runtime = make_temp_runtime(root)
            with ThreadPoolExecutor(max_workers=2) as pool:
                results = list(pool.map(lambda _index: runtime.handle(REQUEST), range(2)))
            approval_ids = [_held_approval_id(result, reused=index == 1) for index, result in enumerate(
                sorted(results, key=lambda item: bool(item.tool_results[0].metadata.get("reused_pending_approval")))
            )]
            pending = runtime.store.list_pending_approvals(limit=100)
            if len(set(approval_ids)) != 1 or len(pending) != 1 or _reminder_rows(reminders_file):
                raise SystemExit(
                    "Concurrent identical reminder requests did not coalesce into one pending approval: "
                    f"ids={approval_ids}, pending={[dict(row) for row in pending]}"
                )


def main() -> None:
    assert_reminder_approval_boundary()
    assert_owner_binding_cannot_redirect()
    assert_concurrent_pending_dedupe()
    print("Reminder approval-boundary smoke passed.")


if __name__ == "__main__":
    main()
