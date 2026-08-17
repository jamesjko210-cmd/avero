from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from jarvis_v2.agent.planner import RuleBasedPlanner
from jarvis_v2.scripts.test_runtime import make_temp_runtime
from jarvis_v2.tools import computer as computer_tools
from jarvis_v2.tools.computer import (
    _oav_action_audit_ready_from_metadata,
    _oav_execution_handoff_ready_from_metadata,
    _oav_final_review_ready_from_metadata,
    _oav_post_run_closure_ready_from_metadata,
    _oav_cycle_ledger_ready_from_metadata,
    _oav_cycle_ledger_token_boundary_ready,
    _oav_cycle_ledger_token_sha256,
    _screen_observation_freshness_ready_from_metadata,
)


def assert_read_only_metadata(metadata: dict, label: str) -> None:
    forbidden_true = [
        "calls_model",
        "calls_external_service",
        "executes_tools",
        "reads_personal_data",
        "reads_private_data",
        "executes_side_effect",
        "external_side_effect",
        "writes_files",
        "writes_database",
        "writes_memory",
        "writes_notes",
        "queues_approval",
        "authorizes_execution",
        "authorizes_completion_claim",
        "approval_granted",
        "requires_approval",
        "observes_screen",
        "takes_screenshot",
        "controls_computer",
        "reads_clipboard",
        "runs_shell",
        "speaks",
        "completes_tasks",
    ]
    violations = [key for key in forbidden_true if metadata.get(key)]
    if violations:
        raise SystemExit(f"{label} should be read-only but marked {violations}: {metadata}")
    if metadata.get("operator_timeboxes_override_priority") is not True:
        raise SystemExit(f"{label} missed operator timebox metadata: {metadata}")
    if metadata.get("stop_times_override_priority") is not True:
        raise SystemExit(f"{label} missed stop-time metadata: {metadata}")


def assert_no_future_authority(metadata: dict, label: str) -> None:
    for key in ("authorizes_execution", "authorizes_completion_claim", "approval_granted"):
        if metadata.get(key) is not False:
            raise SystemExit(f"{label} should not grant future authority via {key}: {metadata}")


def assert_operator_limit_output(output: str, label: str) -> None:
    if "explicit stop times" not in output or "pause commands" not in output:
        raise SystemExit(f"{label} missed operator-limit output.")


def assert_no_local_path_blob(value, label: str) -> None:
    blob = json.dumps(value, sort_keys=True, default=str) if not isinstance(value, str) else value
    for fragment in ("/\x55sers/", "/private/", "/var/folders/", "/tmp/"):
        if fragment in blob:
            raise SystemExit(f"{label} leaked a local path: {blob[:1200]}")


def test_planner_routes_screenshot_and_observe() -> None:
    # Real gap found live 2026-07-09: "take a screenshot" and "observe the screen"
    # (the natural phrasings, with an article) missed the planner's exact-substring
    # triggers ("take screenshot" / "observe screen") and fell through to chat.
    p = RuleBasedPlanner()
    for q in ("take a screenshot", "take screenshot", "screenshot"):
        actions = p.plan(q).actions
        if [a.tool_name for a in actions] != ["screenshot"]:
            raise SystemExit(f"screenshot route missed: {q!r} -> {actions}")
    for q in ("observe the screen", "observe screen", "what is on screen"):
        actions = p.plan(q).actions
        if [a.tool_name for a in actions] != ["observe_screen"]:
            raise SystemExit(f"observe_screen route missed: {q!r} -> {actions}")


def main() -> None:
    test_planner_routes_screenshot_and_observe()
    with TemporaryDirectory(prefix="jarvis-computer-plan-") as temp:
        runtime = make_temp_runtime(Path(temp))
        readiness_cases = [
            "computer readiness: open settings, click battery, and verify the battery panel is visible",
            "computer preflight: send an email update",
        ]
        for case in readiness_cases:
            result = runtime.handle(case)
            print(f"[ok={result.verified}] {case}")
            print(result.response[:2200])
            print()
            if not result.verified:
                raise SystemExit(f"Expected '{case}' to run read-only.")
            required = [
                "Computer-control readiness",
                "Readiness verdict",
                "Preflight checklist",
                "Not ready to operate yet",
                "Safe next command",
                "computer task plan",
            ]
            missing = [item for item in required if item not in result.response]
            if missing:
                raise SystemExit(f"Computer readiness missing expected text: {missing}")
            if "observe the screen" in result.response.lower() and "blocked" not in result.response.lower():
                raise SystemExit("Computer readiness should keep screen observation blocked until approval.")
            assert_read_only_metadata(result.tool_results[0].metadata, "Computer readiness")
            assert_operator_limit_output(result.response, "Computer readiness")
        if "Extra caution" not in runtime.handle(readiness_cases[1]).response:
            raise SystemExit("Sensitive objective did not trigger extra caution.")

        action_packet_cases = [
            (
                "computer action packet: click x 100 y 200 expectation battery settings opens",
                ["Computer action packet", "Approval-ready", "observe act verify action click x 100 y 200", "Screen confidence before action", "screen observation confidence", "minimum confidence before action", "Before approval"],
                True,
            ),
            (
                "desktop action packet: type text hello expectation hello appears in the selected search field",
                ["Computer action packet", "type_text", "observe act verify action type_text", "hello appears"],
                True,
            ),
            (
                "computer action packet: click expectation menu opens",
                ["Computer action packet", "Not ready", "x coordinate", "y coordinate"],
                False,
            ),
        ]
        for case, expected, approval_ready in action_packet_cases:
            result = runtime.handle(case)
            print(f"[ok={result.verified}] {case}")
            print(result.response[:2200])
            print()
            if not result.verified:
                raise SystemExit(f"Expected '{case}' to run read-only.")
            missing = [item for item in expected if item not in result.response]
            if missing:
                raise SystemExit(f"Computer action packet missing expected text: {missing}")
            metadata = result.tool_results[0].metadata
            if metadata.get("approval_ready") is not approval_ready:
                raise SystemExit(f"Computer action packet approval_ready mismatch for {case}.")
            if approval_ready and (metadata.get("observation_required_before_action") is not True or metadata.get("minimum_confidence_before_action") != "medium"):
                raise SystemExit(f"Computer action packet missed observation confidence gate metadata: {metadata}")
            assert_read_only_metadata(metadata, "Computer action packet")
            assert_operator_limit_output(result.response, "Computer action packet")

        bounded_packet = runtime.registry.get("computer_action_packet").handler(
            {
                "action": "click",
                "x": 999999,
                "y": -50,
                "expectation": "target opens " + ("safely " * 80),
            }
        )
        print("[ok] direct bounded computer_action_packet")
        print(bounded_packet.output[:900])
        print()
        if not bounded_packet.ok:
            raise SystemExit("Bounded computer action packet should remain read-only and valid.")
        bounded_metadata = bounded_packet.metadata
        if bounded_metadata.get("x") != 10000 or bounded_metadata.get("y") != 0:
            raise SystemExit(f"Computer action packet did not bound coordinates: {bounded_metadata}")
        if len(bounded_metadata.get("expectation", "")) > 220:
            raise SystemExit("Computer action packet did not bound expectation text.")
        assert_read_only_metadata(bounded_metadata, "Bounded computer action packet")
        assert_operator_limit_output(bounded_packet.output, "Bounded computer action packet")

        path_plan = runtime.registry.get("computer_task_plan").handler(
            {"objective": "open /var/folders/zc/jarvis/private/settings.txt and click battery"}
        )
        print("[ok] direct path-redacted computer_task_plan")
        print(path_plan.output[:900])
        print()
        if not path_plan.ok or path_plan.metadata.get("objective") != "open <local-path>":
            raise SystemExit(f"Computer task plan did not redact path-shaped objective: {path_plan.metadata}")
        assert_no_local_path_blob(path_plan.output, "Path-redacted computer task plan output")
        assert_no_local_path_blob(path_plan.metadata, "Path-redacted computer task plan metadata")
        assert_read_only_metadata(path_plan.metadata, "Path-redacted computer task plan")
        assert_operator_limit_output(path_plan.output, "Path-redacted computer task plan")

        path_action = runtime.registry.get("computer_action_packet").handler(
            {
                "spec": "click x 10 y 20 expectation open /tmp/jarvis-computer-settings.txt",
            }
        )
        print("[ok] direct path-redacted computer_action_packet")
        print(path_action.output[:900])
        print()
        if not path_action.ok or path_action.metadata.get("expectation") != "open <local-path>":
            raise SystemExit(f"Computer action packet did not redact path-shaped expectation: {path_action.metadata}")
        assert_no_local_path_blob(path_action.output, "Path-redacted computer action packet output")
        assert_no_local_path_blob(path_action.metadata, "Path-redacted computer action packet metadata")
        assert_read_only_metadata(path_action.metadata, "Path-redacted computer action packet")
        assert_operator_limit_output(path_action.output, "Path-redacted computer action packet")

        path_receipt = runtime.registry.get("approved_screen_observation_receipt").handler(
            {
                "spec": "expectation battery panel visible; observation battery panel visible in settings; source approved_screenshot; action click x 10 y 20; observation id obs-8; screenshot path /tmp/jarvis-oav.png; captured at 2026-06-09T10:00:00+00:00; redaction reviewed safe; approval 8"
            }
        )
        print("[ok] direct path-redacted approved_screen_observation_receipt")
        print(path_receipt.output[:1000])
        print()
        if not path_receipt.ok or path_receipt.metadata.get("screenshot_path") != "<local-path>":
            raise SystemExit(f"Approved screen observation receipt did not redact path-shaped screenshot path: {path_receipt.metadata}")
        assert_no_local_path_blob(path_receipt.output, "Path-redacted approved screen observation receipt output")
        assert_no_local_path_blob(path_receipt.metadata, "Path-redacted approved screen observation receipt metadata")
        assert_read_only_metadata(path_receipt.metadata, "Path-redacted approved screen observation receipt")
        assert_operator_limit_output(path_receipt.output, "Path-redacted approved screen observation receipt")

        receipt_cases = [
            (
                "approved screen observation receipt: expectation battery panel visible; observation battery panel visible in settings; source approved_screenshot; action click x 100 y 200; observation id obs-7; screenshot path /tmp/jarvis_oav_before.png; captured at 2026-06-09T10:00:00+09:00; redaction reviewed safe; approval 7",
                "APPROVED_SCREEN_OBSERVATION_RECEIPT_READY",
                True,
            ),
            (
                "approved screen observation receipt: expectation battery panel visible; observation battery panel visible in settings; source approved_screenshot; action click x 100 y 200",
                "APPROVED_SCREEN_OBSERVATION_SOURCE_ONLY",
                False,
            ),
        ]
        for case, receipt_state, adapter_ready in receipt_cases:
            result = runtime.handle(case)
            print(f"[ok={result.verified}] {case}")
            print(result.response[:1800])
            print()
            if not result.verified:
                raise SystemExit(f"Expected '{case}' to run read-only.")
            if result.tool_results[0].tool_name != "approved_screen_observation_receipt":
                raise SystemExit(f"Expected '{case}' to route to approved_screen_observation_receipt.")
            metadata = result.tool_results[0].metadata
            if metadata.get("receipt_state") != receipt_state:
                raise SystemExit(f"Screen observation receipt state mismatch: {metadata}")
            if metadata.get("real_screenshot_adapter_ready") is not adapter_ready:
                raise SystemExit(f"Screen observation receipt adapter readiness mismatch: {metadata}")
            if metadata.get("observation_receipt_required_for_real_execution") is not True:
                raise SystemExit(f"Screen observation receipt missed real-execution requirement metadata: {metadata}")
            if "/tmp/" in case:
                if metadata.get("screenshot_path") != "<local-path>":
                    raise SystemExit(f"Screen observation receipt should redact temp screenshot path: {metadata}")
                assert_no_local_path_blob(result.response, "Approved screen observation receipt temp output")
                assert_no_local_path_blob(metadata, "Approved screen observation receipt temp metadata")
            assert_read_only_metadata(metadata, "Approved screen observation receipt")
            assert_operator_limit_output(result.response, "Approved screen observation receipt")

        freshness_cases = [
            (
                "screen observation freshness: expectation battery panel visible; observation battery panel visible in settings; source approved_screenshot; action click x 100 y 200; observation id obs-7; screenshot path /tmp/jarvis_oav_before.png; captured at 2026-06-09T10:00:00+00:00; current time 2026-06-09T10:03:00+00:00; max age 300; redaction reviewed safe; approval 7",
                "SCREEN_OBSERVATION_FRESHNESS_READY",
                True,
                180,
            ),
            (
                "screen observation freshness: expectation battery panel visible; observation battery panel visible in settings; source approved_screenshot; action click x 100 y 200; observation id obs-7; screenshot path /tmp/jarvis_oav_before.png; captured at 2026-06-09T10:00:00+00:00; current time 2026-06-09T10:10:00+00:00; max age 300; redaction reviewed safe; approval 7",
                "SCREEN_OBSERVATION_STALE",
                False,
                600,
            ),
            (
                "screen observation freshness: expectation battery panel visible; observation battery panel visible in settings; source approved_screenshot; action click x 100 y 200; observation id obs-7; screenshot path /tmp/jarvis_oav_before.png; captured at 2026-06-09T10:00:00+00:00; max age 300; redaction reviewed safe; approval 7",
                "SCREEN_OBSERVATION_FRESHNESS_HELD",
                False,
                None,
            ),
        ]
        for case, freshness_state, ready, age_seconds in freshness_cases:
            result = runtime.handle(case)
            print(f"[ok={result.verified}] {case}")
            print(result.response[:2200])
            print()
            if not result.verified:
                raise SystemExit(f"Expected '{case}' to run read-only.")
            if result.tool_results[0].tool_name != "screen_observation_freshness_packet":
                raise SystemExit(f"Expected '{case}' to route to screen_observation_freshness_packet.")
            metadata = result.tool_results[0].metadata
            if metadata.get("freshness_state") != freshness_state:
                raise SystemExit(f"Screen observation freshness state mismatch: {metadata}")
            if metadata.get("freshness_ready") is not ready:
                raise SystemExit(f"Screen observation freshness readiness mismatch: {metadata}")
            if metadata.get("screen_observation_freshness_ready") is not ready:
                raise SystemExit(f"Screen observation freshness production-ready alias mismatch: {metadata}")
            if _screen_observation_freshness_ready_from_metadata(metadata) is not ready:
                raise SystemExit(f"Screen observation freshness production validator mismatch: {metadata}")
            tampered_metadata = dict(metadata)
            tampered_metadata["controls_computer"] = True
            if _screen_observation_freshness_ready_from_metadata(tampered_metadata):
                raise SystemExit(f"Screen observation freshness validator accepted computer-control tampering: {tampered_metadata}")
            if ready:
                tampered_metadata = dict(metadata)
                tampered_metadata["observation_age_seconds"] = int(metadata.get("max_age_seconds") or 0) + 1
                if _screen_observation_freshness_ready_from_metadata(tampered_metadata):
                    raise SystemExit(f"Screen observation freshness validator accepted stale-age tampering: {tampered_metadata}")
                tampered_metadata = dict(metadata)
                tampered_metadata["approved_screen_observation_receipt_metadata"] = dict(metadata.get("approved_screen_observation_receipt_metadata") or {})
                tampered_metadata["approved_screen_observation_receipt_metadata"]["real_screenshot_adapter_ready"] = False
                if _screen_observation_freshness_ready_from_metadata(tampered_metadata):
                    raise SystemExit(f"Screen observation freshness validator accepted receipt-readiness tampering: {tampered_metadata}")
            if metadata.get("fresh_enough_for_one_primitive") is not ready:
                raise SystemExit(f"Screen observation freshness one-primitive flag mismatch: {metadata}")
            if metadata.get("observation_age_seconds") != age_seconds:
                raise SystemExit(f"Screen observation freshness age mismatch: {metadata}")
            if freshness_state == "SCREEN_OBSERVATION_STALE" and metadata.get("stale_observation") is not True:
                raise SystemExit(f"Stale freshness packet missed stale flag: {metadata}")
            if freshness_state == "SCREEN_OBSERVATION_FRESHNESS_HELD" and "current review time" not in metadata.get("missing_freshness_proof", []):
                raise SystemExit(f"Freshness packet without current time did not hold on explicit review time: {metadata}")
            if metadata.get("approved_screen_observation_receipt_metadata", {}).get("real_screenshot_adapter_ready") is not True:
                raise SystemExit(f"Freshness packet did not bind approved receipt metadata: {metadata}")
            if "/tmp/" in case:
                receipt_meta = metadata.get("approved_screen_observation_receipt_metadata", {})
                if receipt_meta.get("screenshot_path") != "<local-path>":
                    raise SystemExit(f"Freshness packet should bind redacted temp screenshot path: {metadata}")
                assert_no_local_path_blob(result.response, "Screen observation freshness temp output")
                assert_no_local_path_blob(metadata, "Screen observation freshness temp metadata")
            assert_read_only_metadata(metadata, "Screen observation freshness packet")
            assert_operator_limit_output(result.response, "Screen observation freshness packet")

        confidence_cases = [
            (
                "screen observation confidence: expectation battery panel visible; observation battery panel visible in settings; source approved_screenshot; action click x 100 y 200",
                ["Screen observation confidence packet", "proof state: OBSERVATION_CONFIDENCE_READY", "confidence: high", "ready for action review: yes", "observation receipt state: APPROVED_SCREEN_OBSERVATION_SOURCE_ONLY", "durable screenshot adapter ready: no"],
                "OBSERVATION_CONFIDENCE_READY",
                True,
            ),
            (
                "screen observation confidence: expectation battery panel visible; observation battery settings visible; source approved_screenshot; action click x 100 y 200",
                ["Screen observation confidence packet", "proof state: PARTIAL_OBSERVATION_CONFIDENCE", "confidence: medium", "ready for action review: no"],
                "PARTIAL_OBSERVATION_CONFIDENCE",
                False,
            ),
            (
                "screen observation confidence: expectation battery panel visible; source manual note; action click x 100 y 200",
                ["Screen observation confidence packet", "proof state: NEEDS_APPROVED_OBSERVATION", "confidence: none", "Manual descriptions, stale screenshots, partial matches, and missing observations are not enough"],
                "NEEDS_APPROVED_OBSERVATION",
                False,
            ),
        ]
        for case, expected, proof_state, ready in confidence_cases:
            result = runtime.handle(case)
            print(f"[ok={result.verified}] {case}")
            print(result.response[:2200])
            print()
            if not result.verified:
                raise SystemExit(f"Expected '{case}' to run read-only.")
            missing = [item for item in expected if item not in result.response]
            if missing:
                raise SystemExit(f"Screen observation confidence packet missing expected text: {missing}")
            metadata = result.tool_results[0].metadata
            if metadata.get("proof_state") != proof_state or metadata.get("ready_for_action_review") is not ready:
                raise SystemExit(f"Screen observation confidence packet metadata mismatch: {metadata}")
            if metadata.get("observation_required_before_action") is not True or metadata.get("minimum_confidence_before_action") != "high":
                raise SystemExit(f"Screen observation confidence packet missed required confidence gate metadata: {metadata}")
            if metadata.get("observation_receipt_required_for_real_execution") is not True:
                raise SystemExit(f"Screen observation confidence packet missed real-execution receipt metadata: {metadata}")
            assert_read_only_metadata(metadata, "Screen observation confidence packet")
            assert_operator_limit_output(result.response, "Screen observation confidence packet")

        vision_prompt_cases = [
            (
                "screen vision prompt preview: expectation battery panel visible; observation battery panel visible in settings; source approved_screenshot; observation id obs-7; screenshot path /tmp/jarvis_oav_before.png; captured at 2026-06-09T10:00:00+09:00; redaction reviewed safe; approval 7",
                True,
                "APPROVED_SCREEN_OBSERVATION_RECEIPT_READY",
            ),
            (
                "oav vision prompt: expectation battery panel visible; observation battery settings maybe visible; source manual note",
                False,
                "NEEDS_APPROVED_SCREEN_OBSERVATION_RECEIPT",
            ),
        ]
        for case, handoff_ready, receipt_state in vision_prompt_cases:
            result = runtime.handle(case)
            print(f"[ok={result.verified}] {case}")
            print(result.response[:2800])
            print()
            if not result.verified:
                raise SystemExit(f"Expected '{case}' to run read-only.")
            if result.tool_results[0].tool_name != "screen_vision_prompt_preview":
                raise SystemExit(f"Expected '{case}' to route to screen_vision_prompt_preview.")
            for expected in [
                "Screen vision prompt preview",
                "Prompt sections:",
                "[system]",
                "[user]",
                "[response_contract]",
                "does not call a model",
                "does not read screenshots or image files",
                "cannot approve, execute, or continue",
                "next safe command",
            ]:
                if expected not in result.response:
                    raise SystemExit(f"Screen vision prompt preview missing expected text {expected}: {result.response}")
            metadata = result.tool_results[0].metadata
            if metadata.get("vision_prompt_preview_ready") is not handoff_ready or metadata.get("model_handoff_ready") is not handoff_ready:
                raise SystemExit(f"Screen vision prompt preview readiness mismatch: {metadata}")
            if metadata.get("receipt_state") != receipt_state:
                raise SystemExit(f"Screen vision prompt preview receipt state mismatch: {metadata}")
            if metadata.get("calls_model") is not False or metadata.get("authorizes_model_call") is not False:
                raise SystemExit(f"Screen vision prompt preview should not call or authorize a model: {metadata}")
            if metadata.get("reads_image_file") is not False:
                raise SystemExit(f"Screen vision prompt preview should not read image files: {metadata}")
            if metadata.get("authorizes_computer_control") is not False or metadata.get("authorizes_approval") is not False:
                raise SystemExit(f"Screen vision prompt preview should not grant control or approval: {metadata}")
            prompt_sections = metadata.get("prompt_sections") or {}
            if sorted(prompt_sections) != ["response_contract", "system", "user"]:
                raise SystemExit(f"Screen vision prompt preview missed prompt sections: {metadata}")
            prompt_handoff = metadata.get("screen_vision_prompt_handoff") or {}
            if metadata.get("screen_vision_prompt_handoff_ready") is not True or prompt_handoff.get("handoff_ready") is not True:
                raise SystemExit(f"Screen vision prompt preview missed structured handoff: {metadata}")
            for field, expected in (
                ("ready_for_operator", True),
                ("state_changed", False),
                ("changed", []),
                ("content_in_handoff", False),
                ("authorizes_execution", False),
                ("authorizes_completion_claim", False),
                ("approval_granted", False),
            ):
                if prompt_handoff.get(field) != expected:
                    raise SystemExit(f"Screen vision prompt handoff missed {field}: {prompt_handoff}")
            for field in ["ready_for_operator", "state_changed", "changed", "content_in_handoff", "boundaries", "next_safe_command"]:
                metadata_key = f"screen_vision_prompt_{field}"
                if metadata.get(metadata_key) != prompt_handoff.get(field):
                    raise SystemExit(f"Screen vision prompt handoff flat parity failed for {field}: {metadata} / {prompt_handoff}")
            if metadata.get("screen_vision_prompt_section_keys") != prompt_handoff.get("prompt_section_keys") or metadata.get("screen_vision_prompt_section_count") != len(prompt_sections):
                raise SystemExit(f"Screen vision prompt section parity failed: {metadata} / {prompt_handoff}")
            boundaries = prompt_handoff.get("boundaries") or {}
            for key in ("calls_model", "reads_image_file", "observes_screen", "takes_screenshot", "controls_computer", "queues_approval", "requires_approval", "executes_tools", "authorizes_model_call"):
                if boundaries.get(key):
                    raise SystemExit(f"Screen vision prompt handoff unsafe boundary {key}: {prompt_handoff}")
            assert_no_local_path_blob(metadata, "Screen vision prompt preview metadata")
            assert_no_future_authority(metadata, "Screen vision prompt preview")
            assert_read_only_metadata(metadata, "Screen vision prompt preview")
            assert_operator_limit_output(result.response, "Screen vision prompt preview")

        fake_screenshot = Path(temp) / "jarvis_oav_before.png"
        fake_screenshot.write_bytes(b"fake approved screenshot bytes")
        vision_review_command = (
            f"screen vision model review: expectation battery panel visible; observation battery panel visible in settings; "
            f"source approved_screenshot; observation id obs-7; screenshot path {fake_screenshot}; "
            "captured at 2026-06-09T10:00:00+09:00; redaction reviewed safe; approval 7; consent=true"
        )
        queued_review = runtime.handle(vision_review_command)
        print(f"[blocked={not queued_review.verified}] {vision_review_command}")
        print(queued_review.response[:1000])
        print()
        if queued_review.verified or "explicit approval required" not in queued_review.response:
            raise SystemExit(f"Screen vision model review should queue PERSONAL_DATA approval: {queued_review.response}")
        pending = runtime.store.list_pending_approvals(limit=20)
        if not pending or pending[-1]["tool_name"] != "screen_vision_model_review_preview":
            raise SystemExit(f"Screen vision model review should queue its own approval: {pending}")

        held_review = runtime.registry.get("screen_vision_model_review_preview").handler(
            {
                "expectation": "battery panel visible",
                "observation": "battery panel visible in settings",
                "source": "approved_screenshot",
                "observation_id": "obs-7",
                "screenshot_path": str(fake_screenshot),
                "captured_at": "2026-06-09T10:00:00+09:00",
                "redaction_review": "reviewed safe",
                "approval_id": "7",
                "consent": "true",
            }
        )
        print("[ok] direct held screen_vision_model_review_preview")
        print(held_review.output[:1200])
        print()
        if held_review.ok or held_review.metadata.get("reads_image_file") or held_review.metadata.get("calls_model"):
            raise SystemExit(f"Held screen vision model review should not read image or call model: {held_review.metadata}")
        if held_review.metadata.get("missing_checks") != ["local vision reviewer configured"]:
            raise SystemExit(f"Held screen vision model review should only miss local reviewer: {held_review.metadata}")
        for key in ("reads_private_data", "reads_personal_data", "reads_image_file", "analyzes_image", "calls_model", "controls_computer", "queues_approval"):
            if held_review.metadata.get(key):
                raise SystemExit(f"Held screen vision model review should not mark {key}: {held_review.metadata}")
        assert_no_future_authority(held_review.metadata, "Held screen vision model review")
        assert_no_local_path_blob(held_review.metadata, "Held screen vision model review")

        original_reviewer = computer_tools._SCREEN_VISION_REVIEWER
        computer_tools._SCREEN_VISION_REVIEWER = lambda image_path, prompt_sections: "LIKELY_VERIFIED: battery panel is visible"  # type: ignore[assignment]
        try:
            approved_review = runtime.registry.get("screen_vision_model_review_preview").handler(
                {
                    "expectation": "battery panel visible",
                    "observation": "battery panel visible in settings",
                    "source": "approved_screenshot",
                    "observation_id": "obs-7",
                    "screenshot_path": str(fake_screenshot),
                    "captured_at": "2026-06-09T10:00:00+09:00",
                    "redaction_review": "reviewed safe",
                    "approval_id": "7",
                    "consent": "true",
                }
            )
        finally:
            computer_tools._SCREEN_VISION_REVIEWER = original_reviewer
        print("[ok] direct approved screen_vision_model_review_preview")
        print(approved_review.output[:1600])
        print()
        if not approved_review.ok or "Screen vision model review preview" not in approved_review.output:
            raise SystemExit(f"Approved screen vision model review should return assessment preview: {approved_review}")
        review_metadata = approved_review.metadata
        review_handoff = review_metadata.get("screen_vision_model_review_handoff") or {}
        if not review_metadata.get("screen_vision_model_review_handoff_ready") or not review_handoff:
            raise SystemExit(f"Approved screen vision model review missed handoff: {review_metadata}")
        if review_handoff.get("handoff_ready") is not True:
            raise SystemExit(f"Approved screen vision model review handoff should mark itself ready: {review_handoff}")
        for key in ("reads_private_data", "reads_personal_data", "reads_image_file", "analyzes_image", "calls_model", "requires_approval"):
            if review_metadata.get(key) is not True:
                raise SystemExit(f"Approved screen vision model review should mark {key}: {review_metadata}")
        for key in ("observes_screen", "takes_screenshot", "controls_computer", "queues_approval", "writes_files", "writes_memory", "writes_notes", "executes_tools"):
            if review_metadata.get(key):
                raise SystemExit(f"Approved screen vision model review unsafe metadata {key}: {review_metadata}")
        for field, expected in (
            ("content_in_handoff", True),
            ("state_changed", False),
            ("changed", []),
            ("authorizes_execution", False),
            ("authorizes_completion_claim", False),
            ("approval_granted", False),
        ):
            if review_metadata.get(field) != expected or review_handoff.get(field) != expected:
                raise SystemExit(f"Approved screen vision model review missed {field}: {review_metadata} / {review_handoff}")
        for field in (
            "ready_for_operator",
            "state_changed",
            "changed",
            "content_in_handoff",
            "boundaries",
            "next_safe_command",
        ):
            metadata_key = f"screen_vision_model_review_{field}"
            if review_metadata.get(metadata_key) != review_handoff.get(field):
                raise SystemExit(
                    f"Approved screen vision model review missed flat handoff alias {metadata_key}: "
                    f"{review_metadata} / {review_handoff}"
                )
        if review_handoff.get("assessment_sha256") != review_metadata.get("assessment_sha256"):
            raise SystemExit(f"Approved screen vision model review hash parity failed: {review_metadata} / {review_handoff}")
        if review_metadata.get("screen_vision_model_review_assessment_sha256") != review_handoff.get("assessment_sha256"):
            raise SystemExit(f"Approved screen vision model review flat hash alias failed: {review_metadata} / {review_handoff}")
        if review_metadata.get("screen_vision_model_review_assessment_chars") != len(review_handoff.get("assessment_preview") or ""):
            raise SystemExit(f"Approved screen vision model review flat char-count alias failed: {review_metadata} / {review_handoff}")
        if review_metadata.get("next_command") != review_handoff.get("next_safe_command"):
            raise SystemExit(f"Approved screen vision model review next command parity failed: {review_metadata} / {review_handoff}")
        boundaries = review_handoff.get("boundaries") or {}
        if boundaries.get("reads_image_file") is not True or boundaries.get("controls_computer") is not False or boundaries.get("approval_granted") is not False:
            raise SystemExit(f"Approved screen vision model review boundaries are wrong: {review_handoff}")
        assert_no_local_path_blob(review_metadata, "Approved screen vision model review metadata")
        assert_no_future_authority(review_metadata, "Approved screen vision model review")
        assert_operator_limit_output(approved_review.output, "Approved screen vision model review")

        proof_cases = [
            (
                "observe act verify proof: action click; x 100; y 200; expectation battery panel visible; observation battery panel visible in settings; source approved_screenshot; after battery panel visible in settings",
                [
                    "Observe-act-verify proof packet",
                    "proof state: OAV_PROOF_READY_WITH_AFTER_VERIFICATION",
                    "primitive ready for approval review: yes",
                    "observation receipt state: APPROVED_SCREEN_OBSERVATION_SOURCE_ONLY",
                    "durable screenshot adapter ready: no",
                    "missing proof: none",
                    "does not grant a session-wide screen-control permission",
                ],
                "OAV_PROOF_READY_WITH_AFTER_VERIFICATION",
                True,
            ),
            (
                "oav proof: action click; x 100; y 200; expectation battery panel visible; observation battery settings visible; source approved_screenshot",
                [
                    "Observe-act-verify proof packet",
                    "proof state: OAV_PROOF_INCOMPLETE",
                    "primitive ready for approval review: no",
                    "fresh approved screen observation",
                    "after-action observation",
                ],
                "OAV_PROOF_INCOMPLETE",
                False,
            ),
            (
                "computer control proof packet: action click; expectation menu opens; observation menu opens; source manual note",
                [
                    "Observe-act-verify proof packet",
                    "proof state: OAV_PROOF_INCOMPLETE",
                    "x coordinate",
                    "y coordinate",
                    "source: manual note",
                ],
                "OAV_PROOF_INCOMPLETE",
                False,
            ),
        ]
        for case, expected, proof_state, primitive_ready in proof_cases:
            result = runtime.handle(case)
            print(f"[ok={result.verified}] {case}")
            print(result.response[:2600])
            print()
            if not result.verified:
                raise SystemExit(f"Expected '{case}' to run read-only.")
            if result.tool_results[0].tool_name != "observe_act_verify_proof_packet":
                raise SystemExit(f"Expected '{case}' to route to observe_act_verify_proof_packet.")
            missing = [item for item in expected if item not in result.response]
            if missing:
                raise SystemExit(f"Observe-act-verify proof packet missing expected text: {missing}")
            metadata = result.tool_results[0].metadata
            if metadata.get("proof_state") != proof_state:
                raise SystemExit(f"OAV proof packet proof state mismatch: {metadata}")
            if metadata.get("primitive_ready_for_approval_review") is not primitive_ready:
                raise SystemExit(f"OAV proof packet readiness mismatch: {metadata}")
            if metadata.get("requires_exact_single_step_approval") is not True:
                raise SystemExit(f"OAV proof packet missed single-step approval requirement: {metadata}")
            if metadata.get("observation_receipt_required_for_real_execution") is not True:
                raise SystemExit(f"OAV proof packet missed real-execution observation receipt metadata: {metadata}")
            assert_read_only_metadata(metadata, "Observe-act-verify proof packet")
            assert_operator_limit_output(result.response, "Observe-act-verify proof packet")

        route_lock_cases = [
            (
                "oav route lock: action click; x 100; y 200; expectation battery panel visible; observation battery panel visible in settings; source approved_screenshot; after battery panel visible in settings",
                "OAV_ROUTE_LOCK_HELD",
                False,
                ["approval readiness evidence", "approval packet evidence", "approval chain proof evidence", "approved rerun verification evidence"],
            ),
            (
                "computer route lock: action click; x 100; y 200; expectation battery panel visible; observation battery panel visible in settings; source approved_screenshot; after battery panel visible in settings; approval readiness reviewed; approval packet reviewed; approval chain proof passed; approved rerun verified",
                "OAV_ROUTE_LOCK_READY_FOR_EXPLICIT_REVIEW",
                True,
                [],
            ),
            (
                "desktop route lock: action click; x 100; y 200; expectation battery panel visible; source manual note; approval readiness reviewed; approval packet reviewed; approval chain proof passed; approved rerun verified",
                "OAV_ROUTE_LOCK_HELD",
                False,
                ["OAV proof packet with after-action verification"],
            ),
        ]
        for case, route_state, unlock_candidate, expected_missing in route_lock_cases:
            result = runtime.handle(case)
            print(f"[ok={result.verified}] {case}")
            print(result.response[:3000])
            print()
            if not result.verified:
                raise SystemExit(f"Expected '{case}' to run read-only.")
            if result.tool_results[0].tool_name != "observe_act_verify_route_lock":
                raise SystemExit(f"Expected '{case}' to route to observe_act_verify_route_lock.")
            for expected in [
                "Observe-act-verify route lock",
                "Route posture:",
                "natural-language computer-control routing: disabled",
                "Route proof queue:",
                "next route required:",
                "approval readiness",
                "approval packet",
                "approval chain proof",
                "verification receipt",
                "explicit route lock review",
            ]:
                if expected not in result.response:
                    raise SystemExit(f"OAV route lock missing expected text {expected}: {result.response}")
            metadata = result.tool_results[0].metadata
            if metadata.get("route_lock_state") != route_state:
                raise SystemExit(f"OAV route lock state mismatch: {metadata}")
            if metadata.get("route_unlock_candidate") is not unlock_candidate:
                raise SystemExit(f"OAV route lock unlock candidate mismatch: {metadata}")
            if metadata.get("natural_language_routing_enabled") is not False or metadata.get("natural_language_computer_control_routing_enabled") is not False:
                raise SystemExit(f"OAV route lock should keep natural-language routing disabled: {metadata}")
            if metadata.get("real_execution_approval_gated") is not True or metadata.get("approval_required_for_real_use") is not True:
                raise SystemExit(f"OAV route lock missed approval-gated real-use metadata: {metadata}")
            route_queue = metadata.get("route_proof_queue") or []
            if not route_queue or metadata.get("route_proof_queue_count") != len(route_queue):
                raise SystemExit(f"OAV route lock missed route proof queue metadata: {metadata}")
            if metadata.get("oav_proof_queue") != route_queue or metadata.get("oav_proof_queue_count") != len(route_queue):
                raise SystemExit(f"OAV route lock missed OAV proof queue aliases: {metadata}")
            if metadata.get("next_route_proof_command") != metadata.get("oav_next_proof_command"):
                raise SystemExit(f"OAV route lock next proof aliases diverged: {metadata}")
            if "next route proof:" in result.response:
                raise SystemExit("OAV route lock should render next route required, not next route proof.")
            if metadata.get("next_route_required_command") != metadata.get("next_route_proof_command"):
                raise SystemExit(f"OAV route lock missed next route required alias: {metadata}")
            if metadata.get("oav_next_required_command") != metadata.get("oav_next_proof_command"):
                raise SystemExit(f"OAV route lock missed OAV next required alias: {metadata}")
            if metadata.get("next_required_command") != metadata.get("next_route_required_command"):
                raise SystemExit(f"OAV route lock generic next required alias diverged: {metadata}")
            for blocker in expected_missing:
                if blocker not in metadata.get("missing_blockers", []):
                    raise SystemExit(f"OAV route lock missed blocker {blocker}: {metadata}")
            if unlock_candidate and metadata.get("missing_blockers") != []:
                raise SystemExit(f"OAV route lock should have no blockers when review-ready: {metadata}")
            assert_read_only_metadata(metadata, "Observe-act-verify route lock")
            assert_operator_limit_output(result.response, "Observe-act-verify route lock")

        approval_bridge_cases = [
            (
                "oav approval bridge: action click; x 100; y 200; expectation battery panel visible; observation battery panel visible in settings; source approved_screenshot; after battery panel visible in settings; approval readiness reviewed; approval packet reviewed; approval chain proof passed; approved rerun verified",
                "OAV_APPROVAL_BRIDGE_HELD",
                False,
                False,
                ["numeric approval id", "numeric approved run id", "numeric verification receipt run id"],
            ),
            (
                "oav approval bridge: action click; x 100; y 200; expectation battery panel visible; observation battery panel visible in settings; source approved_screenshot; after battery panel visible in settings; approval readiness reviewed; approval packet reviewed; approval chain proof passed; approved rerun verified; approval 7; approved run 22; verification receipt 22",
                "OAV_APPROVAL_BRIDGE_READY_FOR_FINAL_REVIEW",
                True,
                True,
                [],
            ),
            (
                "computer approval bridge: action click; x 100; y 200; expectation battery panel visible; observation battery panel visible in settings; source approved_screenshot; after battery panel visible in settings; approval readiness reviewed; approval packet reviewed; approval chain proof passed; approved rerun verified; approval 7; approved run 22; verification receipt 23",
                "OAV_APPROVAL_BRIDGE_HELD",
                False,
                False,
                ["verification receipt must match approved run id"],
            ),
        ]
        for case, bridge_state, final_ready, receipt_binding_ready, expected_missing in approval_bridge_cases:
            result = runtime.handle(case)
            print(f"[ok={result.verified}] {case}")
            print(result.response[:3200])
            print()
            if not result.verified:
                raise SystemExit(f"Expected '{case}' to run read-only.")
            if result.tool_results[0].tool_name != "observe_act_verify_approval_bridge":
                raise SystemExit(f"Expected '{case}' to route to observe_act_verify_approval_bridge.")
            for expected in [
                "Observe-act-verify approval bridge",
                "Primitive binding:",
                "Bridge verdict:",
                "Required command chain:",
                "approval readiness",
                "approval packet",
                "approval chain proof",
                "verification receipt",
                "execution audit gate",
                "does not approve, rerun, observe, click, type, move the mouse",
            ]:
                if expected not in result.response:
                    raise SystemExit(f"OAV approval bridge missing expected text {expected}: {result.response}")
            metadata = result.tool_results[0].metadata
            if metadata.get("bridge_state") != bridge_state:
                raise SystemExit(f"OAV approval bridge state mismatch: {metadata}")
            if metadata.get("final_route_review_ready") is not final_ready:
                raise SystemExit(f"OAV approval bridge final review readiness mismatch: {metadata}")
            if metadata.get("receipt_binding_ready") is not receipt_binding_ready:
                raise SystemExit(f"OAV approval bridge receipt binding mismatch: {metadata}")
            if metadata.get("natural_language_routing_enabled") is not False or metadata.get("natural_language_computer_control_routing_enabled") is not False:
                raise SystemExit(f"OAV approval bridge should keep natural-language routing disabled: {metadata}")
            if metadata.get("real_execution_approval_gated") is not True or metadata.get("approval_required_for_real_use") is not True:
                raise SystemExit(f"OAV approval bridge missed approval-gated real-use metadata: {metadata}")
            commands = metadata.get("required_commands") or []
            if not commands or metadata.get("required_command_count") != len(commands):
                raise SystemExit(f"OAV approval bridge missed required command metadata: {metadata}")
            if metadata.get("oav_bridge_required_commands") != commands or metadata.get("oav_bridge_next_command") != metadata.get("next_command"):
                raise SystemExit(f"OAV approval bridge alias metadata mismatch: {metadata}")
            for blocker in expected_missing:
                if blocker not in metadata.get("missing_blockers", []):
                    raise SystemExit(f"OAV approval bridge missed blocker {blocker}: {metadata}")
            assert_read_only_metadata(metadata, "Observe-act-verify approval bridge")
            assert_operator_limit_output(result.response, "Observe-act-verify approval bridge")

        cockpit_cases = [
            (
                "oav cockpit: action click; x 100; y 200; expectation battery panel visible; observation battery panel visible in settings; source approved_screenshot; after battery panel visible in settings; approval readiness reviewed; approval packet reviewed; approval chain proof passed; approved rerun verified; approval 7; approved run 22; verification receipt 22",
                "OAV_COCKPIT_READY_FOR_FINAL_REVIEW",
                True,
                [],
            ),
            (
                "computer control cockpit: action click; x 100; y 200; expectation battery panel visible; observation battery settings visible; source approved_screenshot",
                "OAV_COCKPIT_HELD",
                False,
                ["after-action observation for verification", "numeric approval id"],
            ),
        ]
        for case, cockpit_state, final_ready, expected_missing in cockpit_cases:
            result = runtime.handle(case)
            print(f"[ok={result.verified}] {case}")
            print(result.response[:3600])
            print()
            if not result.verified:
                raise SystemExit(f"Expected '{case}' to run read-only.")
            if result.tool_results[0].tool_name != "observe_act_verify_cockpit":
                raise SystemExit(f"Expected '{case}' to route to observe_act_verify_cockpit.")
            for expected in [
                "Observe-act-verify cockpit",
                "Cockpit state",
                "proof state",
                "route lock state",
                "bridge state",
                "receipt binding ready",
                "final route review ready",
                "natural-language computer-control routing enabled: no",
                "real execution approval-gated: yes",
                "Required command chain",
                "A ready cockpit is still not approval",
            ]:
                if expected not in result.response:
                    raise SystemExit(f"OAV cockpit missing expected text {expected}: {result.response}")
            metadata = result.tool_results[0].metadata
            if metadata.get("cockpit_state") != cockpit_state:
                raise SystemExit(f"OAV cockpit state mismatch: {metadata}")
            if metadata.get("final_route_review_ready") is not final_ready:
                raise SystemExit(f"OAV cockpit final route review readiness mismatch: {metadata}")
            if metadata.get("natural_language_routing_enabled") is not False or metadata.get("natural_language_computer_control_routing_enabled") is not False:
                raise SystemExit(f"OAV cockpit should keep natural-language routing disabled: {metadata}")
            if metadata.get("real_execution_approval_gated") is not True or metadata.get("approval_required_for_real_use") is not True:
                raise SystemExit(f"OAV cockpit missed approval-gated real-use metadata: {metadata}")
            if not isinstance(metadata.get("proof_metadata"), dict) or not isinstance(metadata.get("route_lock_metadata"), dict) or not isinstance(metadata.get("approval_bridge_metadata"), dict):
                raise SystemExit(f"OAV cockpit missed nested proof metadata: {metadata}")
            for blocker in expected_missing:
                if blocker not in metadata.get("missing_blockers", []):
                    raise SystemExit(f"OAV cockpit missed blocker {blocker}: {metadata}")
            assert_read_only_metadata(metadata, "Observe-act-verify cockpit")
            assert_operator_limit_output(result.response, "Observe-act-verify cockpit")

        direct_cockpit = runtime.registry.get("observe_act_verify_cockpit").handler(
            {
                "action": "click",
                "x": 100,
                "y": 200,
                "expectation": "battery panel visible",
                "observation": "battery panel visible in settings",
                "source": "approved_screenshot",
                "after_observation": "battery panel visible in settings",
                "approval_evidence": "approval readiness reviewed",
                "approval_packet_evidence": "approval packet reviewed",
                "approval_chain_evidence": "approval chain proof passed; approved rerun verified",
                "approval_id": 7,
                "approved_run_id": 22,
                "verification_run_id": 22,
            }
        )
        print("[ok] direct observe_act_verify_cockpit")
        print(direct_cockpit.output[:2200])
        print()
        if not direct_cockpit.ok or direct_cockpit.metadata.get("cockpit_state") != "OAV_COCKPIT_READY_FOR_FINAL_REVIEW":
            raise SystemExit(f"Direct OAV cockpit should be ready for final review: {direct_cockpit.metadata}")
        assert_read_only_metadata(direct_cockpit.metadata, "Direct observe-act-verify cockpit")
        assert_operator_limit_output(direct_cockpit.output, "Direct observe-act-verify cockpit")

        final_review_cases = [
            (
                "oav final review: action click; x 100; y 200; expectation battery panel visible; observation battery panel visible in settings; source approved_screenshot; after battery panel visible in settings; approval readiness reviewed; approval packet reviewed; approval chain proof passed; approved rerun verified; approval 7; approved run 22; verification receipt 22; audit execution audit passed; health execution health passed; review operator reviewed final packet",
                "OAV_FINAL_REVIEW_READY_FOR_HUMAN_DECISION",
                True,
                [],
            ),
            (
                "computer final review: action click; x 100; y 200; expectation battery panel visible; observation battery settings visible; source approved_screenshot",
                "OAV_FINAL_REVIEW_HELD",
                False,
                ["ready OAV cockpit", "execution audit evidence", "execution health evidence", "explicit operator final-review evidence"],
            ),
        ]
        for case, final_state, ready_for_decision, expected_missing in final_review_cases:
            result = runtime.handle(case)
            print(f"[ok={result.verified}] {case}")
            print(result.response[:3600])
            print()
            if not result.verified:
                raise SystemExit(f"Expected '{case}' to run read-only.")
            if result.tool_results[0].tool_name != "observe_act_verify_final_review":
                raise SystemExit(f"Expected '{case}' to route to observe_act_verify_final_review.")
            for expected in [
                "Observe-act-verify final review",
                "Final review state",
                "execution audit evidence",
                "execution health evidence",
                "operator final-review evidence",
                "ready for operator decision",
                "A ready final review packet is still not permission to act",
            ]:
                if expected not in result.response:
                    raise SystemExit(f"OAV final review missing expected text {expected}: {result.response}")
            metadata = result.tool_results[0].metadata
            if metadata.get("final_review_state") != final_state:
                raise SystemExit(f"OAV final review state mismatch: {metadata}")
            if metadata.get("ready_for_human_decision") is not ready_for_decision:
                raise SystemExit(f"OAV final review decision readiness mismatch: {metadata}")
            if metadata.get("oav_final_review_ready") is not ready_for_decision:
                raise SystemExit(f"OAV final review production readiness mismatch: {metadata}")
            if _oav_final_review_ready_from_metadata(metadata) is not ready_for_decision:
                raise SystemExit(f"OAV final review validator mismatch: {metadata}")
            if metadata.get("natural_language_routing_enabled") is not False or metadata.get("natural_language_computer_control_routing_enabled") is not False:
                raise SystemExit(f"OAV final review should keep natural-language routing disabled: {metadata}")
            if metadata.get("action_allowed_now") is not False or metadata.get("computer_control_enabled") is not False:
                raise SystemExit(f"OAV final review should not allow action or enabled computer control: {metadata}")
            if metadata.get("real_execution_approval_gated") is not True or metadata.get("approval_required_for_real_use") is not True:
                raise SystemExit(f"OAV final review missed approval-gated real-use metadata: {metadata}")
            if not isinstance(metadata.get("cockpit_metadata"), dict):
                raise SystemExit(f"OAV final review missed nested cockpit metadata: {metadata}")
            for blocker in expected_missing:
                if blocker not in metadata.get("missing_blockers", []):
                    raise SystemExit(f"OAV final review missed blocker {blocker}: {metadata}")
            assert_read_only_metadata(metadata, "Observe-act-verify final review")
            assert_operator_limit_output(result.response, "Observe-act-verify final review")

        direct_final_review = runtime.registry.get("observe_act_verify_final_review").handler(
            {
                "action": "click",
                "x": 100,
                "y": 200,
                "expectation": "battery panel visible",
                "observation": "battery panel visible in settings",
                "source": "approved_screenshot",
                "after_observation": "battery panel visible in settings",
                "approval_evidence": "approval readiness reviewed",
                "approval_packet_evidence": "approval packet reviewed",
                "approval_chain_evidence": "approval chain proof passed; approved rerun verified",
                "approval_id": 7,
                "approved_run_id": 22,
                "verification_run_id": 22,
                "audit_evidence": "execution audit passed",
                "health_evidence": "execution health passed",
                "operator_review": "operator reviewed final packet",
            }
        )
        print("[ok] direct observe_act_verify_final_review")
        print(direct_final_review.output[:2200])
        print()
        if not direct_final_review.ok or direct_final_review.metadata.get("final_review_state") != "OAV_FINAL_REVIEW_READY_FOR_HUMAN_DECISION":
            raise SystemExit(f"Direct OAV final review should be ready for human decision: {direct_final_review.metadata}")
        if not _oav_final_review_ready_from_metadata(direct_final_review.metadata):
            raise SystemExit(f"Direct OAV final review should pass production validator: {direct_final_review.metadata}")
        tampered_final_review = dict(direct_final_review.metadata)
        tampered_final_review["controls_computer"] = True
        if _oav_final_review_ready_from_metadata(tampered_final_review):
            raise SystemExit("OAV final review validator accepted computer-control authority tampering.")
        tampered_final_review = dict(direct_final_review.metadata)
        tampered_final_review["action_allowed_now"] = True
        if _oav_final_review_ready_from_metadata(tampered_final_review):
            raise SystemExit("OAV final review validator accepted action-allowed tampering.")
        tampered_final_review = dict(direct_final_review.metadata)
        tampered_final_review["ready_for_human_decision"] = "true"
        if _oav_final_review_ready_from_metadata(tampered_final_review):
            raise SystemExit("OAV final review validator accepted truthy-string readiness tampering.")
        tampered_final_review = dict(direct_final_review.metadata)
        tampered_final_review["operator_final_review_evidence_ready"] = False
        if _oav_final_review_ready_from_metadata(tampered_final_review):
            raise SystemExit("OAV final review validator accepted missing operator final-review evidence.")
        tampered_final_review = dict(direct_final_review.metadata)
        tampered_cockpit = dict(tampered_final_review.get("cockpit_metadata") or {})
        tampered_cockpit["final_route_review_ready"] = False
        tampered_final_review["cockpit_metadata"] = tampered_cockpit
        if _oav_final_review_ready_from_metadata(tampered_final_review):
            raise SystemExit("OAV final review validator accepted nested cockpit readiness tampering.")
        tampered_final_review = dict(direct_final_review.metadata)
        tampered_cockpit = dict(tampered_final_review.get("cockpit_metadata") or {})
        tampered_cockpit["computer_control_enabled"] = True
        tampered_final_review["cockpit_metadata"] = tampered_cockpit
        if _oav_final_review_ready_from_metadata(tampered_final_review):
            raise SystemExit("OAV final review validator accepted nested computer-control-enabled tampering.")
        assert_read_only_metadata(direct_final_review.metadata, "Direct observe-act-verify final review")
        assert_operator_limit_output(direct_final_review.output, "Direct observe-act-verify final review")

        action_audit_cases = [
            (
                "oav action audit: action click; x 100; y 200; expectation battery panel visible; observation battery panel visible in settings; source approved_screenshot; after battery panel visible in settings; approval readiness reviewed; approval packet reviewed; approval chain proof passed; approved rerun verified; approval 7; approved run 22; verification receipt 22; audit execution audit passed; health execution health passed; review operator reviewed final packet",
                "OAV_ACTION_AUDIT_READY_FOR_APPROVAL_DECISION",
                True,
                [],
            ),
            (
                "computer action audit: action click; x 100; y 200; expectation battery panel visible; observation battery settings visible; source approved_screenshot",
                "OAV_ACTION_AUDIT_HELD",
                False,
                ["ready OAV final review", "execution audit evidence", "execution health evidence", "explicit operator final-review evidence"],
            ),
        ]
        for case, action_audit_state, ready_for_decision, expected_missing in action_audit_cases:
            result = runtime.handle(case)
            print(f"[ok={result.verified}] {case}")
            print(result.response[:3800])
            print()
            if not result.verified:
                raise SystemExit(f"Expected '{case}' to run read-only.")
            if result.tool_results[0].tool_name != "observe_act_verify_action_audit":
                raise SystemExit(f"Expected '{case}' to route to observe_act_verify_action_audit.")
            for expected in [
                "Observe-act-verify action audit",
                "Action audit state",
                "exact primitive command",
                "computer control currently enabled: no",
                "natural-language computer-control routing enabled: no",
                "real execution approval-gated: yes",
                "action allowed now: no",
                "approval decision required: yes",
                "One-primitive boundary",
                "next required evidence",
                "matching verification receipt",
                "after-action learning packet",
            ]:
                if expected not in result.response:
                    raise SystemExit(f"OAV action audit missing expected text {expected}: {result.response}")
            if "next required proof after approval" in result.response:
                raise SystemExit("OAV action audit should render after-approval recovery as evidence, not proof-first prose.")
            metadata = result.tool_results[0].metadata
            if metadata.get("action_audit_state") != action_audit_state:
                raise SystemExit(f"OAV action audit state mismatch: {metadata}")
            if metadata.get("ready_for_approval_decision") is not ready_for_decision:
                raise SystemExit(f"OAV action audit readiness mismatch: {metadata}")
            if metadata.get("oav_action_audit_ready") is not ready_for_decision:
                raise SystemExit(f"OAV action audit production readiness mismatch: {metadata}")
            if _oav_action_audit_ready_from_metadata(metadata) is not ready_for_decision:
                raise SystemExit(f"OAV action audit validator mismatch: {metadata}")
            if metadata.get("action_allowed_now") is not False or metadata.get("approval_decision_required") is not True:
                raise SystemExit(f"OAV action audit should not allow action without approval decision: {metadata}")
            if metadata.get("natural_language_routing_enabled") is not False or metadata.get("natural_language_computer_control_routing_enabled") is not False:
                raise SystemExit(f"OAV action audit should keep natural-language routing disabled: {metadata}")
            if metadata.get("real_execution_approval_gated") is not True or metadata.get("approval_required_for_real_use") is not True:
                raise SystemExit(f"OAV action audit missed approval-gated real-use metadata: {metadata}")
            if not isinstance(metadata.get("final_review_metadata"), dict):
                raise SystemExit(f"OAV action audit missed nested final-review metadata: {metadata}")
            for blocker in expected_missing:
                if blocker not in metadata.get("missing_blockers", []):
                    raise SystemExit(f"OAV action audit missed blocker {blocker}: {metadata}")
            assert_read_only_metadata(metadata, "Observe-act-verify action audit")
            assert_operator_limit_output(result.response, "Observe-act-verify action audit")

        direct_action_audit = runtime.registry.get("observe_act_verify_action_audit").handler(
            {
                "action": "click",
                "x": 100,
                "y": 200,
                "expectation": "battery panel visible",
                "observation": "battery panel visible in settings",
                "source": "approved_screenshot",
                "after_observation": "battery panel visible in settings",
                "approval_evidence": "approval readiness reviewed",
                "approval_packet_evidence": "approval packet reviewed",
                "approval_chain_evidence": "approval chain proof passed; approved rerun verified",
                "approval_id": 7,
                "approved_run_id": 22,
                "verification_run_id": 22,
                "audit_evidence": "execution audit passed",
                "health_evidence": "execution health passed",
                "operator_review": "operator reviewed final packet",
            }
        )
        print("[ok] direct observe_act_verify_action_audit")
        print(direct_action_audit.output[:2200])
        print()
        if not direct_action_audit.ok or direct_action_audit.metadata.get("action_audit_state") != "OAV_ACTION_AUDIT_READY_FOR_APPROVAL_DECISION":
            raise SystemExit(f"Direct OAV action audit should be ready for approval decision: {direct_action_audit.metadata}")
        if not _oav_action_audit_ready_from_metadata(direct_action_audit.metadata):
            raise SystemExit(f"Direct OAV action audit should pass production validator: {direct_action_audit.metadata}")
        tampered_action_audit = dict(direct_action_audit.metadata)
        tampered_action_audit["action_allowed_now"] = True
        if _oav_action_audit_ready_from_metadata(tampered_action_audit):
            raise SystemExit("OAV action audit validator accepted action-now authority tampering.")
        tampered_action_audit = dict(direct_action_audit.metadata)
        tampered_action_audit["ready_for_approval_decision"] = "true"
        if _oav_action_audit_ready_from_metadata(tampered_action_audit):
            raise SystemExit("OAV action audit validator accepted truthy-string readiness tampering.")
        tampered_action_audit = dict(direct_action_audit.metadata)
        tampered_action_audit["exact_primitive_command"] = "observe act verify action click x <x> y 200 expectation battery panel visible"
        if _oav_action_audit_ready_from_metadata(tampered_action_audit):
            raise SystemExit("OAV action audit validator accepted placeholder primitive command.")
        tampered_action_audit = dict(direct_action_audit.metadata)
        tampered_final_review = dict(tampered_action_audit.get("final_review_metadata") or {})
        tampered_final_review["oav_final_review_ready"] = False
        tampered_action_audit["final_review_metadata"] = tampered_final_review
        if _oav_action_audit_ready_from_metadata(tampered_action_audit):
            raise SystemExit("OAV action audit validator accepted nested final-review readiness tampering.")
        assert_read_only_metadata(direct_action_audit.metadata, "Direct observe-act-verify action audit")
        assert_operator_limit_output(direct_action_audit.output, "Direct observe-act-verify action audit")

        execution_handoff_cases = [
            (
                "oav execution handoff: action click; x 100; y 200; expectation battery panel visible; observation battery panel visible in settings; source approved_screenshot; after battery panel visible in settings; approval readiness reviewed; approval packet reviewed; approval chain proof passed; approved rerun verified; approval 7; approved run 22; verification receipt 22; audit execution audit passed; health execution health passed; review operator reviewed final packet",
                "OAV_EXECUTION_HANDOFF_READY_FOR_APPROVAL_PACKET",
                True,
                [],
            ),
            (
                "computer execution handoff: action click; x 100; y 200; expectation battery panel visible; observation battery settings visible; source approved_screenshot",
                "OAV_EXECUTION_HANDOFF_HELD",
                False,
                ["ready OAV action audit"],
            ),
        ]
        for case, handoff_state, ready_for_decision, expected_missing in execution_handoff_cases:
            result = runtime.handle(case)
            print(f"[ok={result.verified}] {case}")
            print(result.response[:4000])
            print()
            if not result.verified:
                raise SystemExit(f"Expected '{case}' to run read-only.")
            if result.tool_results[0].tool_name != "observe_act_verify_execution_handoff":
                raise SystemExit(f"Expected '{case}' to route to observe_act_verify_execution_handoff.")
            for expected in [
                "Observe-act-verify execution handoff",
                "Handoff state",
                "approval readiness",
                "last-look packet",
                "approval chain proof",
                "Post-run proof required",
                "action allowed now: no",
                "does not approve anything",
                "One-shot execution boundary",
            ]:
                if expected not in result.response:
                    raise SystemExit(f"OAV execution handoff missing expected text {expected}: {result.response}")
            metadata = result.tool_results[0].metadata
            if metadata.get("handoff_state") != handoff_state:
                raise SystemExit(f"OAV execution handoff state mismatch: {metadata}")
            if metadata.get("ready_for_approval_decision") is not ready_for_decision:
                raise SystemExit(f"OAV execution handoff readiness mismatch: {metadata}")
            if metadata.get("oav_execution_handoff_ready") is not ready_for_decision:
                raise SystemExit(f"OAV execution handoff production readiness mismatch: {metadata}")
            if _oav_execution_handoff_ready_from_metadata(metadata) is not ready_for_decision:
                raise SystemExit(f"OAV execution handoff validator mismatch: {metadata}")
            if metadata.get("action_allowed_now") is not False or metadata.get("approval_decision_required") is not True:
                raise SystemExit(f"OAV execution handoff should not allow action without approval decision: {metadata}")
            if metadata.get("natural_language_routing_enabled") is not False or metadata.get("natural_language_computer_control_routing_enabled") is not False:
                raise SystemExit(f"OAV execution handoff should keep natural-language routing disabled: {metadata}")
            if metadata.get("real_execution_approval_gated") is not True or metadata.get("approval_required_for_real_use") is not True:
                raise SystemExit(f"OAV execution handoff missed approval-gated real-use metadata: {metadata}")
            post_run = metadata.get("post_run_commands") or []
            if "verification receipt <approved run id>" not in post_run or "after-action learning packet <approved run id>" not in post_run:
                raise SystemExit(f"OAV execution handoff missed post-run proof commands: {metadata}")
            if not isinstance(metadata.get("action_audit_metadata"), dict):
                raise SystemExit(f"OAV execution handoff missed nested action-audit metadata: {metadata}")
            for blocker in expected_missing:
                if blocker not in metadata.get("missing_blockers", []):
                    raise SystemExit(f"OAV execution handoff missed blocker {blocker}: {metadata}")
            assert_read_only_metadata(metadata, "Observe-act-verify execution handoff")
            assert_operator_limit_output(result.response, "Observe-act-verify execution handoff")

        direct_execution_handoff = runtime.registry.get("observe_act_verify_execution_handoff").handler(
            {
                "action": "click",
                "x": 100,
                "y": 200,
                "expectation": "battery panel visible",
                "observation": "battery panel visible in settings",
                "source": "approved_screenshot",
                "after_observation": "battery panel visible in settings",
                "approval_evidence": "approval readiness reviewed",
                "approval_packet_evidence": "approval packet reviewed",
                "approval_chain_evidence": "approval chain proof passed; approved rerun verified",
                "approval_id": 7,
                "approved_run_id": 22,
                "verification_run_id": 22,
                "audit_evidence": "execution audit passed",
                "health_evidence": "execution health passed",
                "operator_review": "operator reviewed final packet",
            }
        )
        print("[ok] direct observe_act_verify_execution_handoff")
        print(direct_execution_handoff.output[:2200])
        print()
        if not direct_execution_handoff.ok or direct_execution_handoff.metadata.get("handoff_state") != "OAV_EXECUTION_HANDOFF_READY_FOR_APPROVAL_PACKET":
            raise SystemExit(f"Direct OAV execution handoff should be ready for approval packet: {direct_execution_handoff.metadata}")
        if direct_execution_handoff.metadata.get("action_allowed_now") is not False:
            raise SystemExit(f"Direct OAV execution handoff should not allow action: {direct_execution_handoff.metadata}")
        if not _oav_execution_handoff_ready_from_metadata(direct_execution_handoff.metadata):
            raise SystemExit(f"Direct OAV execution handoff should pass production validator: {direct_execution_handoff.metadata}")
        tampered_execution_handoff = dict(direct_execution_handoff.metadata)
        tampered_execution_handoff["action_allowed_now"] = True
        if _oav_execution_handoff_ready_from_metadata(tampered_execution_handoff):
            raise SystemExit("OAV execution handoff validator accepted action-now authority tampering.")
        tampered_execution_handoff = dict(direct_execution_handoff.metadata)
        tampered_execution_handoff["ready_for_approval_decision"] = "true"
        if _oav_execution_handoff_ready_from_metadata(tampered_execution_handoff):
            raise SystemExit("OAV execution handoff validator accepted truthy-string readiness tampering.")
        tampered_execution_handoff = dict(direct_execution_handoff.metadata)
        tampered_execution_handoff["post_run_commands"] = ["verification receipt <approved run id>"]
        if _oav_execution_handoff_ready_from_metadata(tampered_execution_handoff):
            raise SystemExit("OAV execution handoff validator accepted incomplete post-run proof queue.")
        tampered_execution_handoff = dict(direct_execution_handoff.metadata)
        tampered_action_audit = dict(tampered_execution_handoff.get("action_audit_metadata") or {})
        tampered_action_audit["oav_action_audit_ready"] = False
        tampered_execution_handoff["action_audit_metadata"] = tampered_action_audit
        if _oav_execution_handoff_ready_from_metadata(tampered_execution_handoff):
            raise SystemExit("OAV execution handoff validator accepted nested action-audit readiness tampering.")
        assert_read_only_metadata(direct_execution_handoff.metadata, "Direct observe-act-verify execution handoff")
        assert_operator_limit_output(direct_execution_handoff.output, "Direct observe-act-verify execution handoff")

        post_run_closure_cases = [
            (
                "oav post-run closure: action click; x 100; y 200; expectation battery panel visible; observation battery panel visible in settings; source approved_screenshot; after battery panel visible in settings; approval readiness reviewed; approval packet reviewed; approval chain proof passed; approved rerun verified; approval 7; approved run 22; verification receipt 22; audit execution audit passed; health execution health passed; review operator reviewed final packet; verification verification receipt 22 verified; post health execution health passed; post audit execution audit passed; learning after-action learning reviewed",
                "OAV_POST_RUN_CLOSURE_READY_FOR_NEXT_PRIMITIVE_REVIEW",
                True,
                [],
            ),
            (
                "oav post-run closure: action click; x 100; y 200; expectation battery panel visible; observation battery panel visible in settings; source approved_screenshot; after battery panel visible in settings; approval readiness reviewed; approval packet reviewed; approval chain proof passed; approved rerun verified; approval 7; approved run 22; verification receipt 22; audit execution audit passed; health execution health passed; review operator reviewed final packet",
                "OAV_POST_RUN_CLOSURE_HELD",
                False,
                ["after-action learning evidence"],
            ),
        ]
        for case, closure_state, ready_for_next_review, expected_missing in post_run_closure_cases:
            result = runtime.handle(case)
            print(f"[ok={result.verified}] {case}")
            print(result.response[:4000])
            print()
            if not result.verified:
                raise SystemExit(f"Expected '{case}' to run read-only.")
            if result.tool_results[0].tool_name != "observe_act_verify_post_run_closure":
                raise SystemExit(f"Expected '{case}' to route to observe_act_verify_post_run_closure.")
            for expected in [
                "Observe-act-verify post-run closure",
                "Closure state",
                "action allowed now: no",
                "Continuation boundary",
                "fresh observation required before next primitive review: yes",
                "previous primitive approval reusable for next primitive: no",
                "Closure command chain",
            ]:
                if expected not in result.response:
                    raise SystemExit(f"OAV post-run closure missing expected text {expected}: {result.response}")
            metadata = result.tool_results[0].metadata
            if metadata.get("closure_state") != closure_state:
                raise SystemExit(f"OAV post-run closure state mismatch: {metadata}")
            if metadata.get("ready_for_next_primitive_review") is not ready_for_next_review:
                raise SystemExit(f"OAV post-run closure readiness mismatch: {metadata}")
            if metadata.get("oav_post_run_closure_ready") is not ready_for_next_review:
                raise SystemExit(f"OAV post-run closure production readiness mismatch: {metadata}")
            if _oav_post_run_closure_ready_from_metadata(metadata) is not ready_for_next_review:
                raise SystemExit(f"OAV post-run closure validator mismatch: {metadata}")
            if metadata.get("action_allowed_now") is not False:
                raise SystemExit(f"OAV post-run closure should not allow action: {metadata}")
            if metadata.get("natural_language_routing_enabled") is not False or metadata.get("natural_language_computer_control_routing_enabled") is not False:
                raise SystemExit(f"OAV post-run closure should keep natural-language routing disabled: {metadata}")
            if metadata.get("real_execution_approval_gated") is not True or metadata.get("approval_required_for_real_use") is not True:
                raise SystemExit(f"OAV post-run closure missed approval-gated real-use metadata: {metadata}")
            if metadata.get("next_review_requires_fresh_observation") is not True:
                raise SystemExit(f"OAV post-run closure missed fresh-observation next-review gate: {metadata}")
            for key in [
                "previous_approval_reusable_for_next_primitive",
                "previous_observation_reusable_for_next_primitive",
                "previous_verification_reusable_for_next_primitive",
                "previous_approved_run_reusable_for_next_primitive",
            ]:
                if metadata.get(key) is not False:
                    raise SystemExit(f"OAV post-run closure should forbid prior proof reuse for next primitive via {key}: {metadata}")
            for key in [
                "next_review_requires_new_route_lock",
                "next_review_requires_new_approval_bridge",
                "next_review_requires_new_final_review",
                "next_review_requires_new_action_audit",
                "next_review_requires_new_execution_handoff",
                "prior_primitive_proof_only",
            ]:
                if metadata.get(key) is not True:
                    raise SystemExit(f"OAV post-run closure missed next-review proof invariant {key}: {metadata}")
            for blocker in expected_missing:
                if blocker not in metadata.get("missing_blockers", []):
                    raise SystemExit(f"OAV post-run closure missed blocker {blocker}: {metadata}")
            if ready_for_next_review:
                if metadata.get("next_primitive_review_state") != "FRESH_OAV_REVIEW_UNLOCKED":
                    raise SystemExit(f"OAV post-run closure missed unlocked next-review state: {metadata}")
                if not str(metadata.get("next_review_start_command", "")).startswith("observe act verify cockpit:"):
                    raise SystemExit(f"OAV post-run closure missed next-review start command: {metadata}")
                if metadata.get("receipt_binding_ready") is not True:
                    raise SystemExit(f"OAV post-run closure missed receipt binding: {metadata}")
                for key in ["post_run_verification_ready", "post_run_health_ready", "post_run_audit_ready", "after_action_learning_ready"]:
                    if metadata.get(key) is not True:
                        raise SystemExit(f"OAV post-run closure missed ready flag {key}: {metadata}")
                if "ready for next primitive review: yes" not in result.response:
                    raise SystemExit(f"OAV post-run closure missed ready output: {result.response}")
            assert_read_only_metadata(metadata, "Observe-act-verify post-run closure")
            assert_operator_limit_output(result.response, "Observe-act-verify post-run closure")

        direct_post_run_closure = runtime.registry.get("observe_act_verify_post_run_closure").handler(
            {
                "action": "click",
                "x": 100,
                "y": 200,
                "expectation": "battery panel visible",
                "observation": "battery panel visible in settings",
                "source": "approved_screenshot",
                "after_observation": "battery panel visible in settings",
                "approval_evidence": "approval readiness reviewed",
                "approval_packet_evidence": "approval packet reviewed",
                "approval_chain_evidence": "approval chain proof passed; approved rerun verified",
                "approval_id": 7,
                "approved_run_id": 22,
                "verification_run_id": 22,
                "audit_evidence": "execution audit passed",
                "health_evidence": "execution health passed",
                "operator_review": "operator reviewed final packet",
                "post_verification": "verification receipt 22 verified",
                "post_health": "execution health passed",
                "post_audit": "execution audit passed",
                "learning_evidence": "after-action learning reviewed",
            }
        )
        print("[ok] direct observe_act_verify_post_run_closure")
        print(direct_post_run_closure.output[:2200])
        print()
        if not direct_post_run_closure.ok or direct_post_run_closure.metadata.get("closure_state") != "OAV_POST_RUN_CLOSURE_READY_FOR_NEXT_PRIMITIVE_REVIEW":
            raise SystemExit(f"Direct OAV post-run closure should be ready for next primitive review: {direct_post_run_closure.metadata}")
        if direct_post_run_closure.metadata.get("action_allowed_now") is not False:
            raise SystemExit(f"Direct OAV post-run closure should not allow action: {direct_post_run_closure.metadata}")
        if direct_post_run_closure.metadata.get("next_review_requires_fresh_observation") is not True:
            raise SystemExit(f"Direct OAV post-run closure missed fresh next-review gate: {direct_post_run_closure.metadata}")
        if direct_post_run_closure.metadata.get("previous_approval_reusable_for_next_primitive") is not False:
            raise SystemExit(f"Direct OAV post-run closure should forbid prior approval reuse: {direct_post_run_closure.metadata}")
        if not _oav_post_run_closure_ready_from_metadata(direct_post_run_closure.metadata):
            raise SystemExit(f"Direct OAV post-run closure should pass production validator: {direct_post_run_closure.metadata}")
        tampered_post_run_closure = dict(direct_post_run_closure.metadata)
        tampered_post_run_closure["previous_approval_reusable_for_next_primitive"] = True
        if _oav_post_run_closure_ready_from_metadata(tampered_post_run_closure):
            raise SystemExit("OAV post-run closure validator accepted prior approval reuse tampering.")
        tampered_post_run_closure = dict(direct_post_run_closure.metadata)
        tampered_post_run_closure["ready_for_next_primitive_review"] = "true"
        if _oav_post_run_closure_ready_from_metadata(tampered_post_run_closure):
            raise SystemExit("OAV post-run closure validator accepted truthy-string readiness tampering.")
        tampered_post_run_closure = dict(direct_post_run_closure.metadata)
        tampered_post_run_closure["verification_run_id"] = 23
        if _oav_post_run_closure_ready_from_metadata(tampered_post_run_closure):
            raise SystemExit("OAV post-run closure validator accepted mismatched verification receipt.")
        tampered_post_run_closure = dict(direct_post_run_closure.metadata)
        tampered_handoff = dict(tampered_post_run_closure.get("handoff_metadata") or {})
        tampered_handoff["oav_execution_handoff_ready"] = False
        tampered_post_run_closure["handoff_metadata"] = tampered_handoff
        if _oav_post_run_closure_ready_from_metadata(tampered_post_run_closure):
            raise SystemExit("OAV post-run closure validator accepted nested execution-handoff tampering.")
        assert_read_only_metadata(direct_post_run_closure.metadata, "Direct observe-act-verify post-run closure")
        assert_operator_limit_output(direct_post_run_closure.output, "Direct observe-act-verify post-run closure")

        cycle_ledger_cases = [
            (
                "oav cycle ledger: action click; x 100; y 200; expectation battery panel visible; observation battery panel visible in settings; source approved_screenshot; after battery panel visible in settings; approval readiness reviewed; approval packet reviewed; approval chain proof passed; approved rerun verified; approval 7; approved run 22; verification receipt 22; audit execution audit passed; health execution health passed; review operator reviewed final packet; verification verification receipt 22 verified; post health execution health passed; post audit execution audit passed; learning after-action learning reviewed",
                "OAV_CYCLE_LEDGER_READY_FOR_FRESH_REVIEW",
                True,
                [],
            ),
            (
                "oav cycle ledger: action click; x 100; y 200; expectation battery panel visible; observation battery panel visible in settings; source approved_screenshot; after battery panel visible in settings; approval readiness reviewed; approval packet reviewed; approval chain proof passed; approved rerun verified; approval 7; approved run 22; verification receipt 22; audit execution audit passed; health execution health passed; review operator reviewed final packet",
                "OAV_CYCLE_LEDGER_HELD",
                False,
                ["after-action learning evidence", "post_run_closure"],
            ),
        ]
        for case, cycle_state, ready_for_fresh_review, expected_missing in cycle_ledger_cases:
            result = runtime.handle(case)
            print(f"[ok={result.verified}] {case}")
            print(result.response[:4000])
            print()
            if not result.verified:
                raise SystemExit(f"Expected '{case}' to run read-only.")
            if result.tool_results[0].tool_name != "observe_act_verify_cycle_ledger":
                raise SystemExit(f"Expected '{case}' to route to observe_act_verify_cycle_ledger.")
            for expected in [
                "Observe-act-verify cycle ledger",
                "Cycle state",
                "action allowed now: no",
                "Cycle stages",
                "Fresh-review boundary",
                "Fresh-review preflight queue",
                "Fresh-review contract rows",
                "Prior primitive proof cannot authorize a new action",
                "Proof queue",
            ]:
                if expected not in result.response:
                    raise SystemExit(f"OAV cycle ledger missing expected text {expected}: {result.response}")
            metadata = result.tool_results[0].metadata
            if metadata.get("cycle_state") != cycle_state:
                raise SystemExit(f"OAV cycle ledger state mismatch: {metadata}")
            if metadata.get("ready_for_fresh_next_primitive_review") is not ready_for_fresh_review:
                raise SystemExit(f"OAV cycle ledger readiness mismatch: {metadata}")
            if metadata.get("oav_cycle_ledger_ready") is not ready_for_fresh_review:
                raise SystemExit(f"OAV cycle ledger validator flag mismatch: {metadata}")
            if _oav_cycle_ledger_ready_from_metadata(metadata) is not ready_for_fresh_review:
                raise SystemExit(f"OAV cycle ledger production validator mismatch: {metadata}")
            if metadata.get("action_allowed_now") is not False:
                raise SystemExit(f"OAV cycle ledger should not allow action: {metadata}")
            if metadata.get("computer_control_enabled") is not False:
                raise SystemExit(f"OAV cycle ledger should keep computer control disabled: {metadata}")
            if metadata.get("natural_language_computer_control_routing_enabled") is not False:
                raise SystemExit(f"OAV cycle ledger should keep natural-language computer-control routing disabled: {metadata}")
            if metadata.get("real_execution_approval_gated") is not True or metadata.get("approval_required_for_real_use") is not True:
                raise SystemExit(f"OAV cycle ledger missed approval-gated real-use metadata: {metadata}")
            if metadata.get("next_review_requires_fresh_observation") is not True:
                raise SystemExit(f"OAV cycle ledger missed fresh-observation gate: {metadata}")
            for key in [
                "previous_approval_reusable_for_next_primitive",
                "previous_observation_reusable_for_next_primitive",
                "previous_verification_reusable_for_next_primitive",
                "previous_approved_run_reusable_for_next_primitive",
            ]:
                if metadata.get(key) is not False:
                    raise SystemExit(f"OAV cycle ledger should forbid prior proof reuse via {key}: {metadata}")
            for key in [
                "prior_primitive_proof_authorizes_new_action",
                "prior_primitive_proof_authorizes_model_call",
                "prior_primitive_proof_authorizes_tool_execution",
                "prior_primitive_proof_authorizes_computer_control",
                "prior_primitive_proof_authorizes_screenshot",
                "prior_primitive_proof_authorizes_personal_data_read",
                "prior_primitive_proof_authorizes_external_side_effect",
                "prior_primitive_proof_authorizes_approval",
                "prior_primitive_proof_authorizes_route_unlock",
                "prior_primitive_proof_authorizes_verification_shortcut",
            ]:
                if metadata.get(key) is not False:
                    raise SystemExit(f"OAV cycle ledger should keep prior primitive proof non-authorizing via {key}: {metadata}")
            fresh_review_queue = metadata.get("fresh_review_preflight_queue") or []
            if metadata.get("fresh_review_preflight_queue_count") != len(fresh_review_queue):
                raise SystemExit(f"OAV cycle ledger missed fresh-review preflight queue count: {metadata}")
            if metadata.get("fresh_review_next_preflight_command") != "screen observation freshness: <next primitive approved observation>":
                raise SystemExit(f"OAV cycle ledger missed first fresh-review command: {metadata}")
            for expected_command in [
                "screen observation freshness: <next primitive approved observation>",
                "observe act verify proof: action <next primitive>; expectation <expected screen change>; observation <fresh approved observation>; source approved_screenshot",
                "observe act verify route lock: action <next primitive>; approval <new approval id>",
                "observe act verify approval bridge: action <next primitive>; approval <new approval id>; approved run <new approved run id>; verification receipt <new receipt id>",
                "observe act verify final review: action <next primitive>",
                "observe act verify action audit: action <next primitive>",
                "observe act verify execution handoff: action <next primitive>",
                "observe act verify post-run closure: action <next primitive>",
            ]:
                if expected_command not in fresh_review_queue:
                    raise SystemExit(f"OAV cycle ledger missed fresh-review command {expected_command!r}: {metadata}")
            fresh_contract = metadata.get("fresh_review_contract_rows") or []
            expected_contract_items = {
                "fresh_screen_observation",
                "new_route_lock",
                "new_approval_bridge",
                "new_final_review",
                "new_action_audit",
                "new_execution_handoff",
                "new_post_run_closure",
                "fresh_cycle_ledger_review",
            }
            if metadata.get("fresh_review_contract_row_count") != len(fresh_contract) or len(fresh_contract) != 8:
                raise SystemExit(f"OAV cycle ledger missed fresh-review contract rows: {metadata}")
            if {row.get("item") for row in fresh_contract} != expected_contract_items:
                raise SystemExit(f"OAV cycle ledger missed fresh-review contract items: {metadata}")
            if metadata.get("fresh_review_contract_ready") is not True:
                raise SystemExit(f"OAV cycle ledger fresh-review contract should be ready as a non-authorizing contract: {metadata}")
            if any(
                row.get("required") is not True
                or row.get("prior_artifact_reusable") is not False
                or row.get("authorizes_action_now") is not False
                or row.get("authorizes_model_call") is not False
                or row.get("authorizes_tool_execution") is not False
                or row.get("authorizes_computer_control") is not False
                or row.get("authorizes_screenshot") is not False
                or row.get("authorizes_personal_data_read") is not False
                or row.get("authorizes_external_side_effect") is not False
                or row.get("authorizes_approval") is not False
                or row.get("authorizes_route_unlock") is not False
                or row.get("authorizes_verification_shortcut") is not False
                for row in fresh_contract
            ):
                raise SystemExit(f"OAV cycle ledger fresh-review contract rows should be non-authorizing: {metadata}")
            stage_rows = metadata.get("stage_rows") or []
            if metadata.get("stage_count") != len(stage_rows) or len(stage_rows) < 10:
                raise SystemExit(f"OAV cycle ledger missed stage rows: {metadata}")
            if ready_for_fresh_review and not all(row.get("ready") for row in stage_rows):
                raise SystemExit(f"OAV cycle ledger should mark all stages ready: {metadata}")
            if any(
                row.get("reusable_for_next_primitive") is not False
                or row.get("authorizes_action_now") is not False
                or row.get("authorizes_computer_control") is not False
                or row.get("authorizes_screenshot") is not False
                or row.get("authorizes_approval") is not False
                or row.get("authorizes_route_unlock") is not False
                or row.get("authorizes_model_call") is not False
                or row.get("authorizes_tool_execution") is not False
                or row.get("authorizes_personal_data_read") is not False
                or row.get("authorizes_external_side_effect") is not False
                for row in stage_rows
            ):
                raise SystemExit(f"OAV cycle ledger stage rows should be proof-only and non-reusable: {metadata}")
            token = str(metadata.get("oav_cycle_ledger_token_sha256") or "")
            token_rows = metadata.get("oav_cycle_ledger_token_boundary_rows") or []
            if len(token) != 64 or any(char not in "0123456789abcdefABCDEF" for char in token):
                raise SystemExit(f"OAV cycle ledger missed token hash: {metadata}")
            if metadata.get("oav_cycle_ledger_token_present") is not True:
                raise SystemExit(f"OAV cycle ledger missed token presence: {metadata}")
            if metadata.get("oav_cycle_ledger_token_boundary_row_count") != len(token_rows) or len(token_rows) != 4:
                raise SystemExit(f"OAV cycle ledger missed token boundary rows: {metadata}")
            if metadata.get("oav_cycle_ledger_token_boundary_ready") is not True:
                raise SystemExit(f"OAV cycle ledger missed token boundary ready flag: {metadata}")
            if not _oav_cycle_ledger_token_boundary_ready(token, token_rows):
                raise SystemExit(f"OAV cycle ledger token boundary failed production validator: {metadata}")
            if {
                "oav_cycle_ledger_token",
                "fresh_review_contract_boundary",
                "stage_row_boundary",
                "next_primitive_review_boundary",
            } != {row.get("item") for row in token_rows}:
                raise SystemExit(f"OAV cycle ledger token boundary items diverged: {metadata}")
            if any(
                row.get("proof_only") is not True
                or row.get("token_sha256") != token
                or row.get("reusable_for_next_primitive") is not False
                or row.get("authorizes_action_now") is not False
                or row.get("authorizes_computer_control") is not False
                or row.get("authorizes_screenshot") is not False
                or row.get("authorizes_approval") is not False
                or row.get("authorizes_route_unlock") is not False
                or row.get("authorizes_model_call") is not False
                or row.get("authorizes_tool_execution") is not False
                or row.get("authorizes_personal_data_read") is not False
                or row.get("authorizes_external_side_effect") is not False
                for row in token_rows
            ):
                raise SystemExit(f"OAV cycle ledger token boundary should be proof-only and non-authorizing: {metadata}")
            for key in [
                "oav_cycle_ledger_token_authorizes_action_now",
                "oav_cycle_ledger_token_authorizes_computer_control",
                "oav_cycle_ledger_token_authorizes_screenshot",
                "oav_cycle_ledger_token_authorizes_approval",
                "oav_cycle_ledger_token_authorizes_route_unlock",
                "oav_cycle_ledger_token_authorizes_model_call",
                "oav_cycle_ledger_token_authorizes_tool_execution",
                "oav_cycle_ledger_token_authorizes_personal_data_read",
                "oav_cycle_ledger_token_authorizes_external_side_effect",
                "oav_cycle_ledger_token_reusable_for_next_primitive",
            ]:
                if metadata.get(key) is not False:
                    raise SystemExit(f"OAV cycle ledger token should report {key}=False: {metadata}")
            if metadata.get("next_primitive_requires_new_oav_cycle_ledger_token") is not True:
                raise SystemExit(f"OAV cycle ledger missed next-token requirement: {metadata}")
            tampered_token_rows = [dict(row) for row in token_rows]
            tampered_token_rows[0]["authorizes_computer_control"] = True
            if _oav_cycle_ledger_token_boundary_ready(token, tampered_token_rows):
                raise SystemExit(f"OAV cycle ledger token boundary validator should reject authority tampering: {metadata}")
            tampered_token_status_rows = [dict(row) for row in token_rows]
            tampered_token_status_rows[3]["status"] = "fresh_oav_review_optional"
            if _oav_cycle_ledger_token_boundary_ready(token, tampered_token_status_rows):
                raise SystemExit(f"OAV cycle ledger token boundary validator should reject status tampering: {metadata}")
            recomputed_token = _oav_cycle_ledger_token_sha256(
                action=str(metadata.get("action") or ""),
                x=metadata.get("x"),
                y=metadata.get("y"),
                expectation=str(metadata.get("expectation") or ""),
                approved_run_id=metadata.get("approved_run_id"),
                verification_run_id=metadata.get("verification_run_id"),
                cycle_state=str(metadata.get("cycle_state") or ""),
                fresh_review_contract_rows=fresh_contract,
                stage_rows=stage_rows,
            )
            if recomputed_token != token:
                raise SystemExit(f"OAV cycle ledger token should be reproducible: {metadata}")
            tampered_contract = [dict(row) for row in fresh_contract]
            tampered_contract[0]["authorizes_computer_control"] = True
            if _oav_cycle_ledger_token_sha256(
                action=str(metadata.get("action") or ""),
                x=metadata.get("x"),
                y=metadata.get("y"),
                expectation=str(metadata.get("expectation") or ""),
                approved_run_id=metadata.get("approved_run_id"),
                verification_run_id=metadata.get("verification_run_id"),
                cycle_state=str(metadata.get("cycle_state") or ""),
                fresh_review_contract_rows=tampered_contract,
                stage_rows=stage_rows,
            ) == token:
                raise SystemExit(f"OAV cycle ledger token should bind contract authority flags: {metadata}")
            tampered_stages = [dict(row) for row in stage_rows]
            tampered_stages[0]["reusable_for_next_primitive"] = True
            if _oav_cycle_ledger_token_sha256(
                action=str(metadata.get("action") or ""),
                x=metadata.get("x"),
                y=metadata.get("y"),
                expectation=str(metadata.get("expectation") or ""),
                approved_run_id=metadata.get("approved_run_id"),
                verification_run_id=metadata.get("verification_run_id"),
                cycle_state=str(metadata.get("cycle_state") or ""),
                fresh_review_contract_rows=fresh_contract,
                stage_rows=tampered_stages,
            ) == token:
                raise SystemExit(f"OAV cycle ledger token should bind stage reuse flags: {metadata}")
            if metadata.get("required_command_count") != len(metadata.get("required_commands") or []):
                raise SystemExit(f"OAV cycle ledger missed proof queue count: {metadata}")
            for blocker in expected_missing:
                if blocker not in metadata.get("missing_blockers", []):
                    raise SystemExit(f"OAV cycle ledger missed blocker {blocker}: {metadata}")
            assert_read_only_metadata(metadata, "Observe-act-verify cycle ledger")
            assert_operator_limit_output(result.response, "Observe-act-verify cycle ledger")

        direct_cycle_ledger = runtime.registry.get("observe_act_verify_cycle_ledger").handler(
            {
                "action": "click",
                "x": 100,
                "y": 200,
                "expectation": "battery panel visible",
                "observation": "battery panel visible in settings",
                "source": "approved_screenshot",
                "after_observation": "battery panel visible in settings",
                "approval_evidence": "approval readiness reviewed",
                "approval_packet_evidence": "approval packet reviewed",
                "approval_chain_evidence": "approval chain proof passed; approved rerun verified",
                "approval_id": 7,
                "approved_run_id": 22,
                "verification_run_id": 22,
                "audit_evidence": "execution audit passed",
                "health_evidence": "execution health passed",
                "operator_review": "operator reviewed final packet",
                "post_verification": "verification receipt 22 verified",
                "post_health": "execution health passed",
                "post_audit": "execution audit passed",
                "learning_evidence": "after-action learning reviewed",
            }
        )
        print("[ok] direct observe_act_verify_cycle_ledger")
        print(direct_cycle_ledger.output[:2200])
        print()
        if not direct_cycle_ledger.ok or direct_cycle_ledger.metadata.get("cycle_state") != "OAV_CYCLE_LEDGER_READY_FOR_FRESH_REVIEW":
            raise SystemExit(f"Direct OAV cycle ledger should be ready for fresh review: {direct_cycle_ledger.metadata}")
        if direct_cycle_ledger.metadata.get("ready_for_fresh_next_primitive_review") is not True:
            raise SystemExit(f"Direct OAV cycle ledger missed fresh-review readiness: {direct_cycle_ledger.metadata}")
        if direct_cycle_ledger.metadata.get("oav_cycle_ledger_ready") is not True:
            raise SystemExit(f"Direct OAV cycle ledger missed validator-ready flag: {direct_cycle_ledger.metadata}")
        if not _oav_cycle_ledger_ready_from_metadata(direct_cycle_ledger.metadata):
            raise SystemExit(f"Direct OAV cycle ledger failed production validator: {direct_cycle_ledger.metadata}")
        tampered_cycle_ledger = dict(direct_cycle_ledger.metadata)
        tampered_cycle_ledger["ready_for_fresh_next_primitive_review"] = "true"
        if _oav_cycle_ledger_ready_from_metadata(tampered_cycle_ledger):
            raise SystemExit(f"Direct OAV cycle ledger validator should reject truthy-string readiness tampering: {direct_cycle_ledger.metadata}")
        tampered_cycle_ledger = dict(direct_cycle_ledger.metadata)
        tampered_cycle_ledger["stage_rows"] = [dict(row) for row in direct_cycle_ledger.metadata.get("stage_rows", [])]
        tampered_cycle_ledger["stage_rows"][0]["ready"] = False
        if _oav_cycle_ledger_ready_from_metadata(tampered_cycle_ledger):
            raise SystemExit(f"Direct OAV cycle ledger validator should reject stage readiness tampering: {direct_cycle_ledger.metadata}")
        tampered_cycle_ledger = dict(direct_cycle_ledger.metadata)
        tampered_cycle_ledger["fresh_review_contract_rows"] = [dict(row) for row in direct_cycle_ledger.metadata.get("fresh_review_contract_rows", [])]
        tampered_cycle_ledger["fresh_review_contract_rows"][0]["authorizes_tool_execution"] = True
        if _oav_cycle_ledger_ready_from_metadata(tampered_cycle_ledger):
            raise SystemExit(f"Direct OAV cycle ledger validator should reject fresh-review authority tampering: {direct_cycle_ledger.metadata}")
        tampered_cycle_ledger = dict(direct_cycle_ledger.metadata)
        tampered_cycle_ledger["oav_cycle_ledger_token_boundary_rows"] = [dict(row) for row in direct_cycle_ledger.metadata.get("oav_cycle_ledger_token_boundary_rows", [])]
        tampered_cycle_ledger["oav_cycle_ledger_token_boundary_rows"][3]["reusable_for_next_primitive"] = True
        if _oav_cycle_ledger_ready_from_metadata(tampered_cycle_ledger):
            raise SystemExit(f"Direct OAV cycle ledger validator should reject token-boundary reuse tampering: {direct_cycle_ledger.metadata}")
        tampered_cycle_ledger = dict(direct_cycle_ledger.metadata)
        tampered_cycle_ledger["previous_approved_run_reusable_for_next_primitive"] = True
        if _oav_cycle_ledger_ready_from_metadata(tampered_cycle_ledger):
            raise SystemExit(f"Direct OAV cycle ledger validator should reject previous approved-run reuse tampering: {direct_cycle_ledger.metadata}")
        tampered_cycle_ledger = dict(direct_cycle_ledger.metadata)
        tampered_cycle_ledger["post_run_closure_metadata"] = dict(direct_cycle_ledger.metadata.get("post_run_closure_metadata", {}))
        tampered_cycle_ledger["post_run_closure_metadata"]["oav_post_run_closure_ready"] = False
        if _oav_cycle_ledger_ready_from_metadata(tampered_cycle_ledger):
            raise SystemExit(f"Direct OAV cycle ledger validator should reject nested closure tampering: {direct_cycle_ledger.metadata}")
        if direct_cycle_ledger.metadata.get("action_allowed_now") is not False:
            raise SystemExit(f"Direct OAV cycle ledger should not allow action: {direct_cycle_ledger.metadata}")
        assert_read_only_metadata(direct_cycle_ledger.metadata, "Direct observe-act-verify cycle ledger")
        assert_operator_limit_output(direct_cycle_ledger.output, "Direct observe-act-verify cycle ledger")

        verifier_cases = [
            (
                "screen verification contract: expectation battery panel visible; observation battery panel visible in settings; action click battery",
                ["Screen verification contract", "verdict: LIKELY_VERIFIED", "confidence: medium", "matched terms: battery, panel, visible"],
                "LIKELY_VERIFIED",
            ),
            (
                "screen verification contract: expectation battery panel visible; observation settings window visible",
                ["Screen verification contract", "verdict: PARTIAL_MATCH", "missing terms: battery, panel"],
                "PARTIAL_MATCH",
            ),
            (
                "screen verification contract: expectation battery panel visible",
                ["Screen verification contract", "verdict: NEEDS_OBSERVATION", "Observation evidence: <missing>"],
                "NEEDS_OBSERVATION",
            ),
        ]
        for case, expected, verdict in verifier_cases:
            result = runtime.handle(case)
            print(f"[ok={result.verified}] {case}")
            print(result.response[:2200])
            print()
            if not result.verified:
                raise SystemExit(f"Expected '{case}' to run read-only.")
            missing = [item for item in expected if item not in result.response]
            if missing:
                raise SystemExit(f"Screen verification contract missing expected text: {missing}")
            metadata = result.tool_results[0].metadata
            if metadata.get("verdict") != verdict:
                raise SystemExit(f"Screen verification contract verdict mismatch: {metadata}")
            assert_read_only_metadata(metadata, "Screen verification contract")
            assert_operator_limit_output(result.response, "Screen verification contract")

        direct_verifier = runtime.registry.get("screen_verification_contract").handler(
            {
                "expectation": "target opens " + ("safely " * 80),
                "observation": "target opens safely in the expected window",
                "action": "click x 100 y 200",
            }
        )
        print("[ok] direct bounded screen_verification_contract")
        print(direct_verifier.output[:900])
        print()
        if not direct_verifier.ok:
            raise SystemExit("Bounded screen verification contract should remain read-only and valid.")
        if len(direct_verifier.metadata.get("expectation", "")) > 220:
            raise SystemExit("Screen verification contract did not bound expectation text.")
        if direct_verifier.metadata.get("observation_chars", 0) > 600:
            raise SystemExit("Screen verification contract did not bound observation text.")
        assert_read_only_metadata(direct_verifier.metadata, "Bounded screen verification contract")
        assert_operator_limit_output(direct_verifier.output, "Bounded screen verification contract")

        disabled_click = runtime.registry.get("click").handler({"x": 100, "y": 200})
        print("[ok] direct disabled click")
        print(disabled_click.output[:900])
        print()
        if disabled_click.ok:
            raise SystemExit("Disabled click should not run.")
        if disabled_click.metadata.get("controls_computer") or disabled_click.metadata.get("computer_control_enabled") is not False:
            raise SystemExit(f"Disabled click metadata should report no control execution: {disabled_click.metadata}")
        if disabled_click.metadata.get("operator_timeboxes_override_priority") is not True:
            raise SystemExit(f"Disabled click missed operator metadata: {disabled_click.metadata}")
        assert_no_future_authority(disabled_click.metadata, "Disabled click")
        assert_operator_limit_output(disabled_click.output, "Disabled click")

        gated_cases = [
            (
                "observe act verify action click x 100 y 200 expectation battery settings opens",
                "observe_act_verify",
                {"action": "click", "x": 100, "y": 200, "expectation": "battery settings opens"},
            ),
            (
                "observe act verify action type_text text hello expectation hello appears in the selected search field",
                "observe_act_verify",
                {"action": "type_text", "text": "hello", "expectation": "hello appears in the selected search field"},
            ),
            (
                "verify screen expectation battery settings is visible",
                "verify_screen",
                {"expectation": "battery settings is visible"},
            ),
        ]
        approvals_before = len(runtime.store.list_pending_approvals(limit=100))
        for case, tool_name, expected_args in gated_cases:
            result = runtime.handle(case)
            print(f"[blocked={not result.verified}] {case}")
            print(result.response[:1200])
            print()
            if result.verified:
                raise SystemExit(f"Expected '{case}' to be approval-gated.")
            if tool_name not in result.response or "Safety receipt" not in result.response:
                raise SystemExit(f"{case} did not return the expected approval receipt.")
            action = result.plan.actions[0]
            if action.tool_name != tool_name:
                raise SystemExit(f"{case} routed to {action.tool_name}, expected {tool_name}.")
            for key, value in expected_args.items():
                if action.args.get(key) != value:
                    raise SystemExit(f"{case} parsed {key}={action.args.get(key)!r}, expected {value!r}.")
        if len(runtime.store.list_pending_approvals(limit=100)) != approvals_before + len(gated_cases):
            raise SystemExit("Gated computer-control commands did not queue the expected approvals.")
        approvals = runtime.store.list_pending_approvals(limit=10)
        click_approval = next((row for row in approvals if row["user_input"].startswith("observe act verify action click")), None)
        if click_approval is None:
            raise SystemExit("Click OAV approval was not saved.")
        planned_args = json.loads(click_approval["planned_args"])
        if planned_args != {"action": "click", "expectation": "battery settings opens", "x": 100, "y": 200}:
            raise SystemExit(f"Click OAV approval planned args were not preserved: {planned_args!r}")
        detail = runtime.handle(f"approval detail {click_approval['id']}")
        print(f"[ok={detail.verified}] approval detail {click_approval['id']} planned args")
        print(detail.response[:1800])
        print()
        if not detail.verified:
            raise SystemExit("Expected approval detail for OAV approval to run read-only.")
        for expected in ["Planned arguments", "action: click", "x: 100", "y: 200", "expectation: battery settings opens"]:
            if expected not in detail.response:
                raise SystemExit(f"OAV approval detail missed planned arg text: {expected}")
        pending_note = Path(temp) / "Vault" / "Jarvis" / "Automations" / "Pending Approvals.md"
        pending_text = pending_note.read_text(encoding="utf-8")
        for expected in ["Planned arguments", "action: click", "x: 100", "y: 200", "expectation: battery settings opens"]:
            if expected not in pending_text:
                raise SystemExit(f"Pending approvals note missed planned arg text: {expected}")

        cases = [
            "computer task plan: open settings, click battery, and verify the battery panel is visible",
            "observe act verify plan: type hello into the selected search field",
            "computer control plan",
        ]
        for case in cases:
            result = runtime.handle(case)
            print(f"[ok={result.verified}] {case}")
            print(result.response[:2200])
            print()
            if not result.verified:
                raise SystemExit(f"Expected '{case}' to run read-only.")
            required = [
                "Computer task plan",
                "Safety boundary",
                "read-only",
                "Observe-act-verify loop",
                "Stop conditions",
                "explicit approval",
                "observe screen",
            ]
            missing = [item for item in required if item not in result.response]
            if missing:
                raise SystemExit(f"Computer task plan missing expected text: {missing}")
            assert_read_only_metadata(result.tool_results[0].metadata, "Computer task plan")
            assert_operator_limit_output(result.response, "Computer task plan")
        if "type_text" not in runtime.handle(cases[1]).response:
            raise SystemExit("Typing objective did not identify type_text as a likely primitive action.")

        tools = runtime.handle("list tools computer")
        print("[ok] list tools computer")
        print(tools.response[:1200])
        print()
        if "screen_verification_contract" not in tools.response or "screen_observation_freshness_packet" not in tools.response or "screen_observation_confidence_packet" not in tools.response or "screen_vision_prompt_preview" not in tools.response or "screen_vision_model_review_preview" not in tools.response or "observe_act_verify_proof_packet" not in tools.response or "observe_act_verify_route_lock" not in tools.response or "observe_act_verify_approval_bridge" not in tools.response or "observe_act_verify_cockpit" not in tools.response or "observe_act_verify_final_review" not in tools.response or "observe_act_verify_execution_handoff" not in tools.response or "observe_act_verify_post_run_closure" not in tools.response or "observe_act_verify_cycle_ledger" not in tools.response:
            raise SystemExit("Computer tools list missed screen verification/confidence or OAV cycle tools.")


if __name__ == "__main__":
    main()
