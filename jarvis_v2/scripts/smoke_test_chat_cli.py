from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from jarvis_v2.scripts.chat import (
    _chain_proof_summary,
    _approval_packet_succeeded,
    _approval_readiness_succeeded,
    _verification_command_from_chain_proof,
    _verification_receipt_summary,
)


def test_readiness_gate_requires_issued_receipt() -> None:
    ready = SimpleNamespace(
        tool_results=[
            SimpleNamespace(
                tool_name="approval_readiness_packet",
                ok=True,
                metadata={
                    "approval_id": 7,
                    "verdict": "LAST_LOOK_REQUIRED",
                    "approval_readiness_receipt_issued": True,
                },
            )
        ]
    )
    if not _approval_readiness_succeeded(ready, 7):
        raise SystemExit("valid readiness receipt did not unlock approval review")

    for metadata in [
        {
            "approval_id": 7,
            "verdict": "DISMISS_OR_RECONFIRM",
            "approval_readiness_receipt_issued": False,
        },
        {
            "approval_id": 8,
            "verdict": "LAST_LOOK_REQUIRED",
            "approval_readiness_receipt_issued": True,
        },
    ]:
        held = SimpleNamespace(
            tool_results=[
                SimpleNamespace(
                    tool_name="approval_readiness_packet",
                    ok=True,
                    metadata=metadata,
                )
            ]
        )
        if _approval_readiness_succeeded(held, 7):
            raise SystemExit(f"readiness gate accepted an invalid receipt: {metadata}")


def test_packet_gate_requires_viewed_current_readiness() -> None:
    packet = SimpleNamespace(
        tool_results=[
            SimpleNamespace(
                tool_name="approval_execution_packet",
                ok=True,
                metadata={
                    "approval_id": 7,
                    "approval_packet_viewed": True,
                    "approval_readiness_valid": True,
                },
            )
        ]
    )
    if not _approval_packet_succeeded(packet, 7):
        raise SystemExit("valid last-look packet did not unlock the CLI approval prompt")
    packet.tool_results[0].metadata["approval_readiness_valid"] = False
    if _approval_packet_succeeded(packet, 7):
        raise SystemExit("CLI approval prompt accepted a packet without current readiness")


def test_chain_proof_issues_only_a_bounded_verification_command() -> None:
    proof = SimpleNamespace(
        tool_results=[
            SimpleNamespace(
                tool_name="approval_chain_proof",
                ok=True,
                metadata={
                    "approval_id": 7,
                    "verification_command": "verification receipt 42",
                },
            )
        ]
    )
    if _verification_command_from_chain_proof(proof, 7) != "verification receipt 42":
        raise SystemExit("valid chain proof did not issue its bounded verification command")

    invalid_commands = [
        "verification receipt 0",
        "verification receipt -1",
        "verification receipt 42 please",
        "run command something",
    ]
    for command in invalid_commands:
        proof.tool_results[0].metadata["verification_command"] = command
        if _verification_command_from_chain_proof(proof, 7):
            raise SystemExit(f"unbounded chain-proof command was accepted: {command}")

    proof.tool_results[0].metadata.update(
        approval_id=8,
        verification_command="verification receipt 42",
    )
    if _verification_command_from_chain_proof(proof, 7):
        raise SystemExit("verification command from a different approval was accepted")


def test_automatic_proof_summaries_hide_private_fields() -> None:
    hidden_values = ["private request", "private target", "private content", "private output"]
    chain = SimpleNamespace(
        tool_results=[
            SimpleNamespace(
                tool_name="approval_chain_proof",
                ok=True,
                metadata={
                    "approval_id": 7,
                    "verdict": "APPROVAL_CHAIN_PROVEN",
                    "valid_execution_proof": True,
                    "approval_execution_outcome_unknown": False,
                    "linked_runs": 1,
                    "request": hidden_values[0],
                    "target": hidden_values[1],
                    "content": hidden_values[2],
                    "output": hidden_values[3],
                },
            )
        ]
    )
    chain_summary = _chain_proof_summary(chain, 7)
    for value in hidden_values:
        if value in chain_summary:
            raise SystemExit("automatic approval proof summary leaked a private field")
    if "APPROVAL_CHAIN_PROVEN" not in chain_summary or "valid execution proof yes" not in chain_summary:
        raise SystemExit("automatic approval proof summary lost bounded status evidence")

    receipt = SimpleNamespace(
        tool_results=[
            SimpleNamespace(
                tool_name="verification_receipt",
                ok=True,
                metadata={
                    "run_id": 42,
                    "found": True,
                    "ok": True,
                    "approved": True,
                    "verdict": "PASS_WITH_AUDIT_EVIDENCE",
                    "request": hidden_values[0],
                    "target": hidden_values[1],
                    "content": hidden_values[2],
                    "output": hidden_values[3],
                },
            )
        ]
    )
    receipt_summary = _verification_receipt_summary(receipt)
    for value in hidden_values:
        if value in receipt_summary:
            raise SystemExit("automatic verification summary leaked a private field")
    if "PASS_WITH_AUDIT_EVIDENCE" not in receipt_summary or "approval linked yes" not in receipt_summary:
        raise SystemExit("automatic verification summary lost bounded status evidence")


def main() -> None:
    test_readiness_gate_requires_issued_receipt()
    test_packet_gate_requires_viewed_current_readiness()
    test_chain_proof_issues_only_a_bounded_verification_command()
    test_automatic_proof_summaries_hide_private_fields()
    with TemporaryDirectory(prefix="jarvis-chat-cli-") as temp:
        root = Path(temp)
        env = os.environ.copy()
        env.update(
            {
                "JARVIS_DATA_DIR": str(root),
                "JARVIS_DB_PATH": str(root / "jarvis.sqlite"),
                "JARVIS_OBSIDIAN_VAULT": str(root / "Vault"),
                "JARVIS_USE_MODEL_PLANNER": "0",
            }
        )
        result = subprocess.run(
            [sys.executable, "-m", "jarvis_v2.scripts.chat"],
            input="run command python3 --version\ny\n/quit\n",
            text=True,
            capture_output=True,
            timeout=20,
            env=env,
        )
        print(result.stdout)
        if result.stderr:
            print("stderr:")
            print(result.stderr)
        if result.returncode != 0:
            raise SystemExit(f"chat CLI exited with {result.returncode}")
        for expected in [
            "J.A.R.V.I.S. V3 chat online",
            "Enter Jarvis requests only; press Control-C to return to the Terminal shell.",
            "Safety receipt",
            "Readiness: approval readiness 1",
            "Required last look: approval packet 1",
            "Proof check: approval chain proof 1",
            "Approval execution packet #1",
            "Approve approval #1 after this packet?",
            "Approved run result",
            "Automatic approval proof #1:",
            "Automatic verification receipt #",
            "Python",
        ]:
            if expected not in result.stdout:
                raise SystemExit(f"chat CLI approval flow missing expected text: {expected}")
        if "Approve and run this action?" in result.stdout:
            raise SystemExit("chat CLI still used the old direct-approval prompt.")
        ordered = [
            "Approval readiness packet #1:",
            "Approval execution packet #1",
            "Approve approval #1 after this packet?",
            "Approved run result",
            "Automatic approval proof #1:",
            "Automatic verification receipt #",
        ]
        positions = [result.stdout.find(text) for text in ordered]
        if any(position < 0 for position in positions) or positions != sorted(positions):
            raise SystemExit(f"chat CLI approval stages were out of order: {positions}")


if __name__ == "__main__":
    main()
