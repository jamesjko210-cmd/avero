"""Offline checks for bounded dashboard client-side error recovery surfaces."""

from __future__ import annotations

from pathlib import Path


PRIVATE_MARKERS = (
    "/\x55sers/owner/private",
    "JARVIS_STATUS_AUTH_TOKEN=owner-secret",
    "Bearer owner-secret",
    "sqlite:///private/owner.db",
)


def _source_slice(source: str, start: str, end: str) -> str:
    start_index = source.index(start)
    end_index = source.index(end, start_index)
    return source[start_index:end_index]


def test_composer_failure_is_actionable_and_private_safe(source: str) -> None:
    composer = _source_slice(
        source,
        "const submitJarvisMessage = async",
        "const createVoiceConfirmationReceipt = async",
    )
    required = (
        "Check the dashboard connection or sign-in, refresh status",
        "review pending approvals before retrying",
        "do not assume anything ran",
        "Request state is uncertain; order restored",
    )
    if any(fragment not in composer for fragment in required):
        raise SystemExit("dashboard composer recovery guidance drifted")
    forbidden = (
        "data.response || data.error || JSON.stringify(data, null, 2)",
        ": String(error);",
    )
    if any(fragment in composer for fragment in forbidden):
        raise SystemExit("dashboard composer still reflects an unvetted failure payload")
    if any(marker in composer for marker in PRIVATE_MARKERS):
        raise SystemExit("dashboard composer recovery contains a private fixture marker")


def test_approval_failures_reconcile_before_retry(source: str) -> None:
    approvals = _source_slice(
        source,
        "previewApprovalPacketButton?.addEventListener",
        "collapseOutputButton?.addEventListener",
    )
    required = (
        "Nothing was approved.",
        "Do not assume the request ran.",
        "preview the current packet again before retrying",
        "The approval may still be pending.",
        "review the current queue before retrying",
        "!data.ok || typeof data.output !== 'string' || !data.output.trim() || !data.preview_token",
    )
    if any(fragment not in approvals for fragment in required):
        raise SystemExit("dashboard approval recovery guidance drifted")
    forbidden = (
        "throw new Error(data.error",
        "String(error.message || error)",
    )
    if any(fragment in approvals for fragment in forbidden):
        raise SystemExit("dashboard approval recovery still reflects server error detail")
    if any(marker in approvals for marker in PRIVATE_MARKERS):
        raise SystemExit("dashboard approval recovery contains a private fixture marker")


def test_secondary_approval_preview_has_recovery(source: str) -> None:
    preview = _source_slice(
        source,
        "approvalPreviewButtons.forEach",
        "const renderPreflightSummary =",
    )
    required = (
        "Approval preview could not be loaded.",
        "Check the dashboard connection or sign-in, refresh status",
        "Nothing was approved.",
        "if (!data.ok) throw new Error('approval_preview_not_confirmed')",
        "Approval preview returned no displayable packet.",
    )
    if any(fragment not in preview for fragment in required):
        raise SystemExit("secondary approval preview recovery guidance drifted")
    if "approvalPreviewOutput.textContent = String(error)" in preview:
        raise SystemExit("secondary approval preview still reflects browser error detail")
    if "data.output || data.error || JSON.stringify(data, null, 2)" in preview:
        raise SystemExit("secondary approval preview still reflects an unvetted payload")


def test_read_only_workbench_failures_are_private_safe(source: str) -> None:
    preflight = _source_slice(
        source,
        "preflightForm?.addEventListener",
        "const renderActionPacketSummary =",
    )
    action_packet = _source_slice(
        source,
        "actionPacketForm?.addEventListener",
        "const renderBrainSummary =",
    )
    brain = _source_slice(
        source,
        "brainResultCitations?.addEventListener",
        "brainSearchExamples.forEach",
    )
    required = {
        "risk preflight": (
            preflight,
            "throw new Error('risk_preflight_not_confirmed')",
            "Risk preview could not be loaded.",
            "This preview did not execute tools or queue an approval.",
        ),
        "action packet": (
            action_packet,
            "throw new Error('action_packet_not_confirmed')",
            "Action packet could not be loaded.",
            "No computer action ran and no approval was queued.",
        ),
        "brain workbench": (
            brain,
            "throw new Error('memory_detail_not_confirmed')",
            "throw new Error('memory_neighbors_not_confirmed')",
            "throw new Error('brain_result_not_confirmed')",
            "Memory detail could not be loaded.",
            "Related memories could not be loaded.",
            "Brain result could not be loaded.",
            "no memory change was confirmed.",
        ),
    }
    for label, (section, *fragments) in required.items():
        if any(fragment not in section for fragment in fragments):
            raise SystemExit(f"dashboard {label} recovery guidance drifted")
        if "Check the dashboard connection or sign-in, refresh status" not in section:
            raise SystemExit(f"dashboard {label} missed connection/sign-in recovery")
        if any(marker in section for marker in PRIVATE_MARKERS):
            raise SystemExit(f"dashboard {label} recovery contains a private fixture marker")


def test_voice_client_failures_keep_routing_blocked(source: str) -> None:
    voice = _source_slice(
        source,
        "const createVoiceCapturePrivacyPacket = async",
        "chatForm?.addEventListener",
    )
    required = (
        "throw new Error('voice_privacy_packet_not_confirmed')",
        "throw new Error('voice_confirmation_receipt_not_confirmed')",
        "throw new Error('native_voice_start_not_confirmed')",
        "Run `voice setup check` before trying the microphone again.",
        "Voice confirmation was not confirmed.",
        "then create a fresh confirmation receipt.",
        "The order was not routed.",
        "Voice routing blocked; transcript kept until a valid receipt exists.",
    )
    if any(fragment not in voice for fragment in required):
        raise SystemExit("dashboard voice client recovery guidance drifted")
    if any(marker in voice for marker in PRIVATE_MARKERS):
        raise SystemExit("dashboard voice recovery contains a private fixture marker")


def test_dashboard_script_has_no_raw_error_reflection(source: str) -> None:
    script = _source_slice(
        source,
        "const createVoiceCapturePrivacyPacket = async",
        "</script>",
    )
    forbidden = (
        "data.output || data.error || JSON.stringify(data, null, 2)",
        "String(error.message || error)",
        "String(error)",
        "throw new Error(data.error ||",
    )
    found = [fragment for fragment in forbidden if fragment in script]
    if found:
        raise SystemExit(f"dashboard script still reflects raw client errors: {found}")


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    source = (root / "jarvis_v2/ui/status_server.py").read_text(encoding="utf-8")
    test_composer_failure_is_actionable_and_private_safe(source)
    test_approval_failures_reconcile_before_retry(source)
    test_secondary_approval_preview_has_recovery(source)
    test_read_only_workbench_failures_are_private_safe(source)
    test_voice_client_failures_keep_routing_blocked(source)
    test_dashboard_script_has_no_raw_error_reflection(source)
    print(
        "dashboard client error recovery smoke test passed "
        "(composer + approvals + read-only workbenches + voice routing block)"
    )


if __name__ == "__main__":
    main()
