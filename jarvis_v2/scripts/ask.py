from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from jarvis_v2.agent.receipts import runtime_result_receipt
from jarvis_v2.agent.runtime import JarvisRuntime
from jarvis_v2.scripts.startup import is_startup_storage_error, print_startup_failure


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Send one message to Jarvis V3 and print the result.",
        epilog=(
            "Dashboard from the Jarvis V3 project folder: "
            'JARVIS_V3_ENV="$HOME/.jarvis_v3/runtime.env" '
            "./launch_jarvis_v3_dashboard.py. "
            "Use '--' before the message when the Jarvis request itself contains CLI-looking flags."
        ),
    )
    parser.add_argument("message", nargs="*", help="Message to send. If omitted, stdin is used.")
    parser.add_argument("--diagnose", action="store_true", help="Preview routing and approval needs without executing tools.")
    parser.add_argument("--json", action="store_true", help="Print a structured JSON response.")
    parser.add_argument("--strict", action="store_true", help="Exit non-zero if the runtime marks the turn unverified.")
    args, unknown = parser.parse_known_args()

    message = " ".join([*args.message, *unknown]).strip()
    if not message:
        message = sys.stdin.read().strip()
    if not message:
        parser.error("message is required, either as arguments or stdin")

    try:
        runtime = JarvisRuntime()
    except Exception as exc:
        if not is_startup_storage_error(exc):
            raise
        print_startup_failure(exc, json_output=args.json, program="Jarvis ask")
        raise SystemExit(3) from None
    if args.diagnose:
        diagnosis = runtime.diagnose_command(message)
        if args.json:
            print(json.dumps(diagnosis, indent=2, sort_keys=True))
        else:
            _print_diagnosis(diagnosis)
        return

    result = runtime.handle(message)
    if args.json:
        pending_approvals = len(runtime.store.list_pending_approvals(limit=100))
        print(json.dumps(runtime_result_receipt(result, pending_approvals), indent=2, sort_keys=True))
    else:
        print(result.response)
    if args.strict and not result.verified:
        raise SystemExit(2)


def _print_diagnosis(diagnosis: dict[str, Any]) -> None:
    print("Jarvis command diagnosis:")
    print(f"- request: {diagnosis['request']}")
    print(f"- route: {diagnosis['route']}")
    print(f"- recommendation: {diagnosis['recommendation']}")
    print(f"- approval required: {'yes' if diagnosis['approval_required'] else 'no'}")
    print(f"- next command: {diagnosis['next_command']}")
    if diagnosis.get("recommended_next_commands"):
        print("- recommended next commands:")
        for command in diagnosis["recommended_next_commands"]:
            print(f"  - {command}")


if __name__ == "__main__":
    main()
