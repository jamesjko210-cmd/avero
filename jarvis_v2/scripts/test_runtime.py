from __future__ import annotations

from pathlib import Path

from jarvis_v2.agent.runtime import JarvisRuntime
from jarvis_v2.agent.types import RuntimeResult, ToolResult
from jarvis_v2.config import JarvisConfig


def make_temp_runtime(root: Path) -> JarvisRuntime:
    config = JarvisConfig(
        data_dir=root,
        db_path=root / "jarvis.sqlite",
        obsidian_vault=root / "Vault",
        obsidian_root="Jarvis",
        use_model_planner=False,
        watched_dirs=(root / "Watched",),
    )
    return JarvisRuntime(config)


def review_pending_runtime_approval(runtime: JarvisRuntime, approval_id: int) -> ToolResult:
    readiness = runtime.registry.get("approval_readiness_packet").handler({"approval_id": approval_id})
    if not readiness.ok or readiness.metadata.get("approval_readiness_receipt_issued") is not True:
        raise AssertionError(f"Approval readiness setup failed for #{approval_id}: {readiness.output}")

    packet = runtime.registry.get("approval_execution_packet").handler({"approval_id": approval_id})
    if not packet.ok:
        raise AssertionError(f"Approval packet setup failed for #{approval_id}: {packet.output}")
    expected_proof_chain = [
        f"approval readiness {approval_id}",
        f"approval packet {approval_id}",
        f"approve approval {approval_id}",
        f"approval chain proof {approval_id}",
        f"verification receipt <approved run id from approval chain proof {approval_id}>",
    ]
    if packet.metadata.get("proof_chain_commands") != expected_proof_chain:
        raise AssertionError(f"Approval packet returned the wrong proof chain for #{approval_id}: {packet.metadata}")
    return packet


def approve_pending_runtime_approval(runtime: JarvisRuntime, approval_id: int) -> ToolResult:
    review_pending_runtime_approval(runtime, approval_id)
    transition = runtime.registry.get("approve_pending_approval").handler({"approval_id": approval_id})
    if not transition.ok:
        raise AssertionError(f"Approval transition failed for #{approval_id}: {transition.output}")
    if transition.metadata.get("approved_approval_id") != approval_id:
        raise AssertionError(f"Approval transition returned the wrong approval ID: {transition.metadata}")
    return transition


def handle_runtime_case(runtime: JarvisRuntime, user_input: str, *, approved: bool = False) -> RuntimeResult:
    if not approved:
        return runtime.handle(user_input)

    held = runtime.handle(user_input)
    approval_ids = {
        result.metadata.get("approval_id")
        for result in held.tool_results
        if result.metadata.get("requires_confirmation")
        and isinstance(result.metadata.get("approval_id"), int)
        and not isinstance(result.metadata.get("approval_id"), bool)
    }
    if len(approval_ids) != 1:
        raise AssertionError(
            f"Approved runtime smoke case must queue exactly one real approval before rerun: {user_input!r} -> {approval_ids}"
        )
    approval_id = approval_ids.pop()

    transition = approve_pending_runtime_approval(runtime, approval_id)
    rerun_user_input = transition.metadata.get("rerun_user_input")
    if not isinstance(rerun_user_input, str) or rerun_user_input != user_input:
        raise AssertionError(f"Approval transition changed the stored request text: {transition.metadata}")

    return runtime.handle(rerun_user_input, approved=True, approved_approval_id=approval_id)
