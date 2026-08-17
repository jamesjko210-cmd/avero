"""Pin the classify_request word-boundary fix.

RISKY_HINTS in jarvis_v2/tools/autonomy.py matched hints with naive substring
containment (`hint in text`). The "computer" hint list includes "app" (meant
for "open the app"), which is also a substring of "approve"/"approval". So
the moment ANY approval was pending, the only commands that could clear the
queue -- "approve approval N" and "dismiss approval N" -- self-blocked: they
contain the literal word "approval", got flagged as a fabricated "computer"
risk, and execution_governor_packet's has_pending_risky_gate held them
because a "risky" request arrived while approvals were already queued. That
made the dashboard's Approve/Dismiss buttons (and the equivalent typed
commands) permanently inert once one approval existed -- a deadlock with no
in-band recovery. Fixed 2026-07-04 by matching hints on word boundaries.
"""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from jarvis_v2.scripts.test_runtime import make_temp_runtime
from jarvis_v2.tools.autonomy import RISKY_HINTS, _hint_matches


def test_approve_and_dismiss_commands_are_not_flagged_risky() -> None:
    for message in ("approve approval 13", "dismiss approval 13", "approve 13", "dismiss 13"):
        low = message.lower()
        matched = [label for label, hints in RISKY_HINTS.items() if any(_hint_matches(h, low) for h in hints)]
        if matched:
            raise SystemExit(
                f"{message!r} must not be flagged risky (word 'approval' contains 'app' by accident): {matched}"
            )


def test_genuinely_risky_requests_are_still_caught() -> None:
    cases = {
        "open the app": "computer",
        "send a message to mom": "personal data",
        "delete that file": "files",
        "run this script": "shell/code",
    }
    for message, expected_label in cases.items():
        low = message.lower()
        matched = [label for label, hints in RISKY_HINTS.items() if any(_hint_matches(h, low) for h in hints)]
        if expected_label not in matched:
            raise SystemExit(f"{message!r} should still match {expected_label!r}, got {matched}")


def test_dismiss_actually_clears_a_pending_approval_end_to_end() -> None:
    with TemporaryDirectory(prefix="jarvis-approval-governor-") as tmp:
        runtime = make_temp_runtime(Path(tmp))
        approval_id = runtime.store.add_pending_approval(
            session_id=runtime.session_id,
            user_input="send someone a telegram saying hello",
            tool_name="send_telegram",
            reason="Sends a real message to a real person.",
            planned_args={"to": "someone", "message": "hello"},
        )
        before = runtime.store.list_pending_approvals(limit=10)
        if not before:
            raise SystemExit("setup failed: seeded approval is not pending")

        result = runtime.handle(f"dismiss approval {approval_id}")
        after = runtime.store.list_pending_approvals(limit=10)
        if after:
            raise SystemExit(
                f"dismiss approval {approval_id} did not clear the queue: {result.response!r}, still pending: {after}"
            )


def test_governor_packet_does_not_hold_approve_or_dismiss_while_one_is_pending() -> None:
    with TemporaryDirectory(prefix="jarvis-approval-governor-") as tmp:
        runtime = make_temp_runtime(Path(tmp))
        approval_id = runtime.store.add_pending_approval(
            session_id=runtime.session_id,
            user_input="send someone a telegram saying hello",
            tool_name="send_telegram",
            reason="Sends a real message to a real person.",
            planned_args={"to": "someone", "message": "hello"},
        )
        tool = runtime.registry.get("execution_governor_packet")
        for message in (f"approve approval {approval_id}", f"dismiss approval {approval_id}"):
            result = tool.handler({"request": message})
            verdict = result.metadata.get("governor_verdict")
            if verdict == "HOLD_FOR_APPROVAL_REVIEW":
                raise SystemExit(
                    f"execution_governor_packet must not self-block {message!r} with a pending approval present: {result.metadata}"
                )


def main() -> None:
    test_approve_and_dismiss_commands_are_not_flagged_risky()
    test_genuinely_risky_requests_are_still_caught()
    test_dismiss_actually_clears_a_pending_approval_end_to_end()
    test_governor_packet_does_not_hold_approve_or_dismiss_while_one_is_pending()
    print("Approval governor word-boundary smoke passed")


if __name__ == "__main__":
    main()
