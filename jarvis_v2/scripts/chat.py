from __future__ import annotations

import argparse
import re

from jarvis_v2.agent.runtime import JarvisRuntime
from jarvis_v2.scripts.startup import is_startup_storage_error, print_startup_failure
from jarvis_v2.tools.voice import speak_text


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Jarvis V3 chat loop.")
    parser.add_argument("--speak", action="store_true", help="Speak Jarvis responses aloud with macOS text-to-speech.")
    parser.add_argument("--voice", default="", help="Optional macOS voice name for --speak.")
    parser.add_argument("--rate", type=int, default=None, help="Optional speech rate for --speak.")
    args = parser.parse_args()

    try:
        runtime = JarvisRuntime()
    except Exception as exc:
        if not is_startup_storage_error(exc):
            raise
        print_startup_failure(exc, program="Jarvis chat")
        raise SystemExit(3) from exc
    print("J.A.R.V.I.S. V3 chat online. Type /quit to exit.")
    print("Enter Jarvis requests only; press Control-C to return to the Terminal shell.")
    print(f"Model: {runtime.config.chat_model}")
    if args.speak:
        print("Voice output: on")
    print()

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_input:
            continue
        if user_input.lower() in {"/quit", "/exit", "quit", "exit"}:
            break

        result = runtime.handle(user_input)
        print(f"Jarvis: {result.response}")
        if args.speak:
            spoken = speak_text(result.response, voice=args.voice, rate=args.rate, wait=False)
            if not spoken.ok:
                print(f"[voice] {spoken.output}")
        if any(item.metadata.get("requires_confirmation") for item in result.tool_results):
            for approval_id in _approval_ids_from_result(result):
                readiness_result = runtime.handle(f"approval readiness {approval_id}")
                print(f"Jarvis: {readiness_result.response}")
                if not _approval_readiness_succeeded(readiness_result, approval_id):
                    continue
                packet_result = runtime.handle(f"approval packet {approval_id}")
                print(f"Jarvis: {packet_result.response}")
                if not _approval_packet_succeeded(packet_result, approval_id):
                    continue
                approval = input(f"Approve approval #{approval_id} after this packet? [y/N]: ").strip().lower()
                if approval in {"y", "yes"}:
                    approved_result = runtime.handle(f"approve approval {approval_id}")
                    print(f"Jarvis: {approved_result.response}")
                    if args.speak:
                        spoken = speak_text(approved_result.response, voice=args.voice, rate=args.rate, wait=False)
                        if not spoken.ok:
                            print(f"[voice] {spoken.output}")
                    chain_result = runtime.handle(f"approval chain proof {approval_id}")
                    print(f"Jarvis: {_chain_proof_summary(chain_result, approval_id)}")
                    verification_command = _verification_command_from_chain_proof(
                        chain_result,
                        approval_id,
                    )
                    if verification_command:
                        verification_result = runtime.handle(verification_command)
                        print(f"Jarvis: {_verification_receipt_summary(verification_result)}")
        print()


def _approval_ids_from_result(result) -> list[int]:
    approval_ids: list[int] = []
    for item in result.tool_results:
        approval_id = item.metadata.get("approval_id")
        if approval_id is None:
            continue
        try:
            approval_ids.append(int(approval_id))
        except (TypeError, ValueError):
            continue
    return approval_ids


def _approval_readiness_succeeded(result, approval_id: int) -> bool:
    for item in result.tool_results:
        metadata = item.metadata
        if (
            item.tool_name == "approval_readiness_packet"
            and item.ok
            and metadata.get("approval_id") == approval_id
            and metadata.get("verdict") == "LAST_LOOK_REQUIRED"
            and metadata.get("approval_readiness_receipt_issued") is True
        ):
            return True
    return False


def _approval_packet_succeeded(result, approval_id: int) -> bool:
    for item in result.tool_results:
        metadata = item.metadata
        if (
            item.tool_name == "approval_execution_packet"
            and item.ok
            and metadata.get("approval_id") == approval_id
            and metadata.get("approval_packet_viewed") is True
            and metadata.get("approval_readiness_valid") is True
        ):
            return True
    return False


def _verification_command_from_chain_proof(result, approval_id: int) -> str:
    """Return only a bounded receipt command issued by the matching chain proof."""
    for item in result.tool_results:
        metadata = item.metadata
        if (
            item.tool_name != "approval_chain_proof"
            or not item.ok
            or metadata.get("approval_id") != approval_id
        ):
            continue
        command = metadata.get("verification_command")
        if isinstance(command, str) and re.fullmatch(r"verification receipt [1-9][0-9]*", command):
            return command
    return ""


def _safe_receipt_verdict(value: object) -> str:
    if isinstance(value, str) and re.fullmatch(r"[A-Z][A-Z0-9_]{0,79}", value):
        return value
    return "UNREADABLE"


def _chain_proof_summary(result, approval_id: int) -> str:
    """Render content-free approval proof status for the interactive terminal."""
    for item in result.tool_results:
        metadata = item.metadata
        if item.tool_name != "approval_chain_proof" or metadata.get("approval_id") != approval_id:
            continue
        verdict = _safe_receipt_verdict(metadata.get("verdict"))
        valid = metadata.get("valid_execution_proof") is True
        unknown = metadata.get("approval_execution_outcome_unknown") is True
        linked = metadata.get("linked_runs")
        linked_count = linked if isinstance(linked, int) and not isinstance(linked, bool) and linked >= 0 else 0
        return (
            f"Automatic approval proof #{approval_id}: verdict {verdict}; "
            f"valid execution proof {'yes' if valid else 'no'}; "
            f"linked reruns {linked_count}; outcome unknown {'yes' if unknown else 'no'}. "
            "Request, target, content, and raw output are hidden in this automatic summary."
        )
    return (
        f"Automatic approval proof #{approval_id}: UNREADABLE. "
        "Request, target, content, and raw output are hidden in this automatic summary."
    )


def _verification_receipt_summary(result) -> str:
    """Render content-free verification status without replaying stored tool output."""
    for item in result.tool_results:
        if item.tool_name != "verification_receipt":
            continue
        metadata = item.metadata
        run_id = metadata.get("run_id")
        safe_run_id = run_id if isinstance(run_id, int) and not isinstance(run_id, bool) and run_id > 0 else "unknown"
        verdict = _safe_receipt_verdict(metadata.get("verdict"))
        found = metadata.get("found") is True
        ok = metadata.get("ok") is True
        approved = metadata.get("approved") is True
        return (
            f"Automatic verification receipt #{safe_run_id}: verdict {verdict}; "
            f"audit row {'found' if found else 'not found'}; result {'ok' if ok else 'failed or held'}; "
            f"approval linked {'yes' if approved else 'no'}. "
            "Request, target, content, and raw output are hidden in this automatic summary."
        )
    return (
        "Automatic verification receipt: UNREADABLE. "
        "Request, target, content, and raw output are hidden in this automatic summary."
    )


if __name__ == "__main__":
    main()
