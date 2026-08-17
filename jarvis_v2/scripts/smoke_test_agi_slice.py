from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from jarvis_v2.scripts.test_runtime import handle_runtime_case, make_temp_runtime


READ_ONLY_FALSE_FLAGS = [
    "calls_model",
    "executes_tools",
    "queues_approval",
    "approves_request",
    "dismisses_request",
    "writes_memory",
    "writes_files",
    "writes_notes",
    "reads_private_data",
    "reads_personal_data",
    "executes_side_effect",
    "external_side_effect",
    "controls_computer",
    "speaks",
]


def assert_read_only(metadata: dict, label: str) -> None:
    for key in READ_ONLY_FALSE_FLAGS:
        if metadata.get(key) is not False:
            raise SystemExit(f"{label} should be read-only and non-authorizing for {key}: {metadata}")


def assert_review_only_authority(metadata: dict, label: str) -> None:
    expected = {
        "draft_only": True,
        "requires_manual_send": True,
        "loads_without_execution": True,
        "authorizes_execution": False,
        "authorizes_completion_claim": False,
        "approval_granted": False,
    }
    for key, value in expected.items():
        if metadata.get(key) is not value:
            raise SystemExit(f"{label} review-only authority field {key} drifted: {metadata}")


def assert_nested_review_only_handoff(metadata: dict, handoff_name: str, source: str, label: str) -> None:
    handoff = metadata.get(handoff_name) or {}
    if handoff.get("source") != source:
        raise SystemExit(f"{label} missed nested handoff source {source!r}: {metadata}")
    for key in [
        "draft_only",
        "requires_manual_send",
        "loads_without_execution",
        "authorizes_execution",
        "authorizes_completion_claim",
        "approval_granted",
    ]:
        if handoff.get(key) != metadata.get(key):
            raise SystemExit(f"{label} nested handoff authority field {key} diverged: {metadata}")
    assert_review_only_authority(handoff, f"{label} nested handoff")
    for key in READ_ONLY_FALSE_FLAGS:
        if handoff.get(key) is not False:
            raise SystemExit(f"{label} nested handoff should be read-only for {key}: {metadata}")


def assert_safety_flags(metadata: dict, expected: dict[str, bool], label: str) -> None:
    for key, value in expected.items():
        if bool(metadata.get(key, False)) is not value:
            raise SystemExit(f"{label} safety flag {key} drifted; expected {value}: {metadata}")


def main() -> None:
    with TemporaryDirectory(prefix="jarvis-agi-slice-") as temp:
        runtime = make_temp_runtime(Path(temp))
        gate_result = runtime.handle("agi gates")
        print(f"[{'ok' if gate_result.verified else 'blocked'}] agi gates")
        print(gate_result.response[:1800])
        print()
        if not gate_result.verified:
            raise SystemExit("agi gates should run read-only.")
        for expected in [
            "Jarvis AGI-direction gate report",
            "does not claim Jarvis is AGI",
            "Evidence closure commands",
            "Focused verification",
            "Selected next build target",
            "AGI-direction work must stay preview-first",
        ]:
            if expected not in gate_result.response:
                raise SystemExit(f"agi gates missed north-star contract text {expected!r}: {gate_result.response}")
        gate_metadata = gate_result.tool_results[0].metadata
        assert_read_only(gate_metadata, "agi gates")
        assert_review_only_authority(gate_metadata, "agi gates")
        assert_nested_review_only_handoff(gate_metadata, "agi_gate_handoff", "agi_gate_report", "agi gates")
        if gate_metadata.get("gates", 0) < 6 or gate_metadata.get("real_execution_gap_count", 0) < 1:
            raise SystemExit(f"agi gates should expose gate and gap counts: {gate_metadata}")
        if gate_metadata.get("agi_next_build_command") != "agi next build move: long-running autonomy":
            raise SystemExit(f"agi gates should expose the selected next build command: {gate_metadata}")
        if gate_metadata.get("agi_next_gate") != "long-running autonomy":
            raise SystemExit(f"agi gates should expose the selected next gate: {gate_metadata}")
        if not gate_metadata.get("agi_next_target_title"):
            raise SystemExit(f"agi gates missed selected target title: {gate_metadata}")
        if gate_metadata.get("agi_next_likely_file_count") != len(gate_metadata.get("agi_next_likely_files", [])):
            raise SystemExit(f"agi gates selected likely file count diverged: {gate_metadata}")
        if gate_metadata.get("agi_next_acceptance_check_count") != len(gate_metadata.get("agi_next_acceptance_checks", [])):
            raise SystemExit(f"agi gates selected acceptance count diverged: {gate_metadata}")
        if gate_metadata.get("agi_next_focused_verification_command_count") != len(gate_metadata.get("agi_next_focused_verification_commands", [])):
            raise SystemExit(f"agi gates selected focused verification count diverged: {gate_metadata}")
        expected_agi_ready = bool(gate_metadata.get("agi_next_target_files_exist") and gate_metadata.get("agi_next_focused_verification_commands") and gate_metadata.get("agi_next_acceptance_checks"))
        if gate_metadata.get("agi_next_build_packet_ready_for_review") is not expected_agi_ready:
            raise SystemExit(f"agi gates selected build readiness diverged: {gate_metadata}")
        focused_by_gate = gate_metadata.get("focused_verification_by_gate") or {}
        closure_by_gate = gate_metadata.get("evidence_closure_commands_by_gate") or {}
        for gate, commands in focused_by_gate.items():
            if not commands or not all(str(command).startswith("python3 -m ") for command in commands):
                raise SystemExit(f"agi gates focused verification should be concrete Python smoke commands for {gate}: {gate_metadata}")
            if gate not in closure_by_gate or not any(str(command).startswith("completion claim gate: improve AGI gate") for command in closure_by_gate[gate]):
                raise SystemExit(f"agi gates should bind every gate to completion-claim closure commands: {gate_metadata}")

        next_cases = ["agi next build move", "agi next build move: personal integrations"]
        for case in next_cases:
            result = runtime.handle(case)
            print(f"[{'ok' if result.verified else 'blocked'}] {case}")
            print(result.response[:1800])
            print()
            if not result.verified:
                raise SystemExit(f"{case} should run read-only.")
            for expected in [
                "Jarvis AGI next build move",
                "Selected gate:",
                "Focused verification:",
                "Build packet readiness:",
                "Completion proof handoff:",
                "completion claim gate",
                "This packet is planning only.",
            ]:
                if expected not in result.response:
                    raise SystemExit(f"{case} missed expected AGI build packet text {expected!r}: {result.response}")
            metadata = result.tool_results[0].metadata
            assert_read_only(metadata, case)
            assert_review_only_authority(metadata, case)
            assert_nested_review_only_handoff(metadata, "agi_next_build_handoff", "agi_next_build_move", case)
            if not metadata.get("selected_gate") or not metadata.get("next_command"):
                raise SystemExit(f"{case} missed selected gate or next command metadata: {metadata}")
            if not metadata.get("target_title"):
                raise SystemExit(f"{case} missed target title metadata: {metadata}")
            if metadata.get("likely_file_count") != len(metadata.get("likely_files", [])):
                raise SystemExit(f"{case} likely file count diverged: {metadata}")
            if metadata.get("focused_test_count") != len(metadata.get("focused_tests", [])):
                raise SystemExit(f"{case} focused test count diverged: {metadata}")
            if metadata.get("acceptance_check_count") != len(metadata.get("acceptance_checks", [])):
                raise SystemExit(f"{case} acceptance check count diverged: {metadata}")
            if bool(metadata.get("target_integrity_blocks_start")) is metadata.get("target_files_exist"):
                raise SystemExit(f"{case} target integrity block flag should invert target_files_exist: {metadata}")
            expected_ready = bool(metadata.get("target_files_exist") and metadata.get("focused_tests") and metadata.get("acceptance_checks"))
            if metadata.get("build_packet_ready_for_review") is not expected_ready:
                raise SystemExit(f"{case} build-packet readiness flag diverged: {metadata}")
            expected_preflight_ready = bool(
                expected_ready
                and metadata.get("pending_approvals", 0) == 0
                and not metadata.get("execution_health_recovery_closure_blocks_completion_claim")
                and not metadata.get("execution_learning_blocks_completion_claim")
            )
            if metadata.get("implementation_preflight_ready") is not expected_preflight_ready:
                raise SystemExit(f"{case} implementation preflight readiness drifted: {metadata}")
            if metadata.get("agi_next_implementation_preflight_ready") is not metadata.get("implementation_preflight_ready"):
                raise SystemExit(f"{case} AGI implementation preflight alias diverged: {metadata}")
            if metadata.get("implementation_preflight_blocker_count") != len(metadata.get("implementation_preflight_blockers", [])):
                raise SystemExit(f"{case} implementation preflight blocker count diverged: {metadata}")
            if metadata.get("agi_next_implementation_preflight_blocker_count") != metadata.get("implementation_preflight_blocker_count"):
                raise SystemExit(f"{case} AGI implementation preflight blocker alias diverged: {metadata}")
            if metadata.get("agi_next_implementation_preflight_blockers") != metadata.get("implementation_preflight_blockers"):
                raise SystemExit(f"{case} AGI implementation preflight blocker list diverged: {metadata}")
            if not metadata.get("implementation_preflight_ready") and not metadata.get("implementation_preflight_next_command"):
                raise SystemExit(f"{case} blocked implementation preflight should expose the next command: {metadata}")
            if metadata.get("agi_next_implementation_preflight_next_command") != metadata.get("implementation_preflight_next_command"):
                raise SystemExit(f"{case} AGI implementation preflight next-command alias diverged: {metadata}")
            if metadata.get("focused_verification_commands") != metadata.get("focused_tests"):
                raise SystemExit(f"{case} focused verification aliases diverged: {metadata}")
            if metadata.get("focused_verification_command_count") != len(metadata.get("focused_verification_commands", [])):
                raise SystemExit(f"{case} focused verification count diverged: {metadata}")
            if not any(str(command).startswith("completion claim gate: improve AGI gate") for command in metadata.get("evidence_closure_commands", [])):
                raise SystemExit(f"{case} missed completion-claim evidence closure command: {metadata}")
            if "personal integrations" in case and metadata.get("selected_gate") != "personal integrations":
                raise SystemExit(f"{case} should honor the requested AGI gate: {metadata}")

        cases = [
            ("integration status", False, True, "integration_status", {"writes_files": False, "writes_notes": False, "reads_personal_data": False, "executes_side_effect": False, "queues_approval": False}),
            ("jarvis doctor", False, True, "jarvis_doctor", {"writes_files": False, "writes_notes": False, "reads_personal_data": False, "executes_side_effect": False, "queues_approval": False}),
            ("computer control status", False, True, "computer_control_status", {"writes_files": False, "writes_notes": False, "controls_computer": False, "executes_side_effect": False, "queues_approval": False}),
            ("weak memories", False, True, "list_weak_memories", {"writes_memory": False, "writes_files": False, "writes_notes": False, "reads_personal_data": False, "queues_approval": False}),
            ("update memory tree snapshot", False, True, "memory_tree_summary", {"writes_memory": False, "writes_files": False, "writes_notes": False, "reads_personal_data": False, "queues_approval": False}),
            ("daily brief", False, True, "daily_brief", {"writes_files": True, "writes_notes": True, "reads_personal_data": False, "external_side_effect": False, "queues_approval": False}),
            ("list tools", False, True, "list_tools", {"writes_files": False, "writes_notes": False, "reads_personal_data": False, "controls_computer": False, "queues_approval": False}),
            ("remind me to test high risk approval", False, False, "create_reminder", {}),
        ]
        for case, approved, should_verify, expected_tool, expected_flags in cases:
            result = handle_runtime_case(runtime, case, approved=approved)
            status = "ok" if result.verified else "blocked"
            print(f"[{status}] {case}")
            print(result.response[:1600])
            print()
            if result.verified is not should_verify:
                raise SystemExit(f"{case} verification state drifted: expected {should_verify}, got {result.verified}")
            tool_names = [tool_result.tool_name for tool_result in result.tool_results]
            if expected_tool not in tool_names:
                raise SystemExit(f"{case} routed to unexpected tool(s): {tool_names}")
            metadata = result.tool_results[0].metadata if result.tool_results else {}
            if should_verify:
                assert_safety_flags(metadata, expected_flags, case)
            else:
                if not metadata.get("requires_confirmation"):
                    raise SystemExit(f"{case} should stay approval-gated before execution: {metadata}")
                if "explicit approval required" not in result.response or "Safety receipt" not in result.response:
                    raise SystemExit(f"{case} should surface approval receipt guidance: {result.response}")


if __name__ == "__main__":
    main()
