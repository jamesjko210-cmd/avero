from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from jarvis_v2.agent.planner import RuleBasedPlanner
from jarvis_v2.scripts.test_runtime import make_temp_runtime


def _assert_rehearsal_handoff(metadata: dict, *, key: str, source: str, include_context: bool = False) -> None:
    handoff = metadata.get(key) or {}
    if handoff.get("source") != source:
        raise SystemExit(f"{source} handoff missed source: {handoff}")
    if metadata.get(f"{source}_handoff_ready") is not True or handoff.get("handoff_ready") is not True:
        raise SystemExit(f"{source} handoff missed readiness aliases: {handoff} vs {metadata}")
    expected_flat = {
        "ready_for_operator": True,
        "state_changed": False,
        "changed": [],
        "content_in_handoff": False,
        "next_safe_command": metadata.get("next_command"),
        "next_safe_commands": metadata.get("recommended_next_commands"),
        "next_safe_command_count": len(metadata.get("recommended_next_commands") or []),
        "authorizes_execution": False,
        "authorizes_completion_claim": False,
        "approval_granted": False,
    }
    for field, expected_value in expected_flat.items():
        if handoff.get(field) != expected_value:
            raise SystemExit(f"{source} handoff {field} diverged: {handoff} vs {metadata}")
        if metadata.get(f"{source}_{field}") != expected_value:
            raise SystemExit(f"{source} flat alias {field} diverged: {handoff} vs {metadata}")
    for field in ["mode", "route", "recommendation", "next_command", "safe_to_execute_now", "approval_required"]:
        if handoff.get(field) != metadata.get(field):
            raise SystemExit(f"{source} handoff diverged for {field}: {handoff} vs {metadata}")
    if handoff.get("recommended_next_commands") != metadata.get("recommended_next_commands"):
        raise SystemExit(f"{source} handoff missed recommended commands: {handoff}")
    if handoff.get("recommended_next_command_count") != len(metadata.get("recommended_next_commands") or []):
        raise SystemExit(f"{source} handoff missed recommended command count: {handoff}")
    planned = metadata.get("planned_actions") or []
    if handoff.get("planned_action_count") != len(planned):
        raise SystemExit(f"{source} handoff missed planned action count: {handoff}")
    if handoff.get("planned_tools") != [str(action.get("tool") or "") for action in planned]:
        raise SystemExit(f"{source} handoff missed planned tools: {handoff}")
    if handoff.get("risk_counts") != metadata.get("risk_counts"):
        raise SystemExit(f"{source} handoff missed risk counts: {handoff}")
    if handoff.get("recovery_closure_state") != metadata.get("recovery_closure_state"):
        raise SystemExit(f"{source} handoff missed recovery state: {handoff}")
    if handoff.get("recovery_closure_proof_queue_count") != metadata.get("recovery_closure_proof_queue_count"):
        raise SystemExit(f"{source} handoff missed recovery queue count: {handoff}")
    if handoff.get("execution_learning_state") != metadata.get("execution_learning_state"):
        raise SystemExit(f"{source} handoff missed learning state: {handoff}")
    if handoff.get("execution_learning_proof_queue_count") != metadata.get("execution_learning_proof_queue_count"):
        raise SystemExit(f"{source} handoff missed learning queue count: {handoff}")
    if include_context and handoff.get("context") != metadata.get("context"):
        raise SystemExit(f"{source} handoff missed context counts: {handoff}")
    expected_false = [
        "calls_model",
        "calls_chat_model",
        "executes_tools",
        "queues_approval",
        "approves_request",
        "dismisses_request",
        "reads_private_data",
        "reads_personal_data",
        "executes_side_effect",
        "external_side_effect",
        "writes_files",
        "writes_notes",
        "writes_memory",
        "controls_computer",
        "requires_approval",
        "speaks",
        "authorizes_execution",
        "authorizes_completion_claim",
        "approval_granted",
    ]
    for field in expected_false:
        if handoff.get(field) is not False:
            raise SystemExit(f"{source} handoff unsafe {field}: {handoff}")
    for value in [handoff.get("next_command"), handoff.get("recommended_next_commands"), handoff.get("planned_tools")]:
        text = str(value)
        if any(fragment in text for fragment in ["/\x55sers/", "/private/", "/var/folders/", "/tmp/"]):
            raise SystemExit(f"{source} handoff leaked local path text: {handoff}")


def _assert_command_diagnosis_handoff(metadata: dict) -> None:
    handoff = metadata.get("command_diagnosis_handoff") or {}
    if handoff.get("source") != "command_diagnosis":
        raise SystemExit(f"Command diagnosis handoff missed source: {handoff}")
    for field in [
        "route",
        "recommendation",
        "next_command",
        "approval_required",
        "pending_approvals",
        "forecast_new_approvals",
        "forecast_reused_approval_ids",
        "safe_to_execute_now",
        "recovery_closure_state",
        "recovery_closure_blocks_auto_execution",
        "recovery_closure_proof_queue_count",
        "recovery_closure_next_proof_command",
        "execution_learning_state",
        "execution_learning_blocks_completion_claim",
        "execution_learning_proof_queue_count",
        "execution_learning_next_proof_command",
    ]:
        if handoff.get(field) != metadata.get(field):
            raise SystemExit(f"Command diagnosis handoff diverged for {field}: {handoff} vs {metadata}")
    if handoff.get("recommended_next_commands") != metadata.get("recommended_next_commands"):
        raise SystemExit(f"Command diagnosis handoff missed recommended commands: {handoff}")
    if handoff.get("recommended_next_command_count") != len(metadata.get("recommended_next_commands") or []):
        raise SystemExit(f"Command diagnosis handoff missed recommended command count: {handoff}")
    planned = metadata.get("planned_actions") or []
    if handoff.get("planned_action_count") != len(planned):
        raise SystemExit(f"Command diagnosis handoff missed planned action count: {handoff}")
    if handoff.get("planned_tools") != [str(item.get("tool") or "") for item in planned]:
        raise SystemExit(f"Command diagnosis handoff missed planned tools: {handoff}")
    if handoff.get("approval_forecast_count") != len(metadata.get("approval_queue_forecast") or []):
        raise SystemExit(f"Command diagnosis handoff missed approval forecast count: {handoff}")
    if handoff.get("forecast_queue_before") != metadata.get("forecast_queue_before"):
        raise SystemExit(f"Command diagnosis handoff missed forecast queue before: {handoff}")
    if handoff.get("forecast_queue_after_if_sent") != metadata.get("forecast_queue_after_if_sent"):
        raise SystemExit(f"Command diagnosis handoff missed forecast queue after: {handoff}")
    expected_false = [
        "calls_model",
        "executes_tools",
        "queues_approval",
        "approves_request",
        "dismisses_request",
        "reads_private_data",
        "reads_personal_data",
        "executes_side_effect",
        "writes_files",
        "writes_memory",
        "writes_notes",
        "controls_computer",
        "authorizes_execution",
        "authorizes_completion_claim",
        "approval_granted",
    ]
    for field in expected_false:
        if handoff.get(field) is not False:
            raise SystemExit(f"Command diagnosis handoff unsafe {field}: {handoff}")
    for value in [handoff.get("next_command"), handoff.get("recommended_next_commands"), handoff.get("planned_tools")]:
        text = str(value)
        if any(fragment in text for fragment in ["/\x55sers/", "/private/", "/var/folders/", "/tmp/"]):
            raise SystemExit(f"Command diagnosis handoff leaked local path text: {handoff}")


def main() -> None:
    planner = RuleBasedPlanner()
    route_cases = {
        "assistant turn rehearsal please": ("assistant_turn_rehearsal", {"message": "what should Jarvis do next"}),
        "assistant rehearsal please": ("assistant_turn_rehearsal", {"message": "what should Jarvis do next"}),
        "chat rehearsal please": ("assistant_turn_rehearsal", {"message": "what should Jarvis do next"}),
        "message rehearsal please": ("assistant_turn_rehearsal", {"message": "what should Jarvis do next"}),
        "show assistant turn rehearsal": ("assistant_turn_rehearsal", {"message": "what should Jarvis do next"}),
        "show assistant rehearsal": ("assistant_turn_rehearsal", {"message": "what should Jarvis do next"}),
        "show latest assistant turn rehearsal": ("assistant_turn_rehearsal", {"message": "what should Jarvis do next"}),
        "show latest chat rehearsal": ("assistant_turn_rehearsal", {"message": "what should Jarvis do next"}),
        "preview assistant turn rehearsal": ("assistant_turn_rehearsal", {"message": "what should Jarvis do next"}),
        "preview assistant turn": ("assistant_turn_rehearsal", {"message": "what should Jarvis do next"}),
        "preview chat turn": ("assistant_turn_rehearsal", {"message": "what should Jarvis do next"}),
        "assistant turn preview please": ("assistant_turn_rehearsal", {"message": "what should Jarvis do next"}),
        "chat turn preview please": ("assistant_turn_rehearsal", {"message": "what should Jarvis do next"}),
        "rehearse assistant turn please": ("assistant_turn_rehearsal", {"message": "what should Jarvis do next"}),
        "dry run assistant turn please": ("assistant_turn_rehearsal", {"message": "what should Jarvis do next"}),
        "chat dry run please": ("assistant_turn_rehearsal", {"message": "what should Jarvis do next"}),
        "preview assistant turn: can we talk about memory please": (
            "assistant_turn_rehearsal",
            {"message": "can we talk about memory please"},
        ),
        "rehearse assistant turn: can we talk about memory please": (
            "assistant_turn_rehearsal",
            {"message": "can we talk about memory please"},
        ),
        "assistant turn rehearsal: can we talk about memory please": (
            "assistant_turn_rehearsal",
            {"message": "can we talk about memory please"},
        ),
        "rehearse: can we talk about memory please": (
            "action_rehearsal",
            {"request": "can we talk about memory please"},
        ),
    }
    for command, (expected_tool, expected_args) in route_cases.items():
        plan = planner.plan(command)
        actual = [(action.tool_name, action.args) for action in plan.actions]
        if actual != [(expected_tool, expected_args)]:
            raise SystemExit(f"Assistant turn rehearsal planner route mismatch for {command!r}: {actual}")

    with TemporaryDirectory(prefix="jarvis-turn-rehearsal-") as temp:
        runtime = make_temp_runtime(Path(temp))
        runtime.handle("set preference response style to direct and warm category communication")
        runtime.handle("remember that Jarvis should preview assistant turns before risky execution")

        cases = [
            (
                "assistant turn rehearsal: can we talk about Jarvis memory?",
                [
                    "Jarvis assistant turn rehearsal",
                    "Routing preview",
                    "mode: chat",
                    "would answer conversationally",
                    "Context preview",
                    "active preferences",
                    "Safety boundary",
                    "does not call the chat model",
                    "does not",
                    "queue approvals",
                    "Next command",
                    "send this message normally",
                ],
            ),
            (
                "assistant turn rehearsal: run command python3 --version",
                [
                    "Jarvis assistant turn rehearsal",
                    "mode: tool",
                    "run_shell_command",
                    "HIGH_RISK",
                    "approval required",
                    "approval readiness #ID",
                    "approval packet #ID",
                    "approval chain proof #ID",
                    "ToolRegistry",
                    "PermissionPolicy",
                    "queue approvals",
                    "Next command",
                    "execution governor: run command python3 --version",
                ],
            ),
            (
                "chat rehearsal: remember that previews should not write memory",
                [
                    "Jarvis assistant turn rehearsal",
                    "mode: tool",
                    "remember",
                    "LOCAL_SAFE",
                    "would be allowed by default",
                    "does not",
                    "save memory",
                    "Next command",
                    "execution governor: remember that previews should not write memory",
                ],
            ),
            (
                "assistant turn rehearsal: continue building Jarvis V2 as an agent harness",
                [
                    "Jarvis assistant turn rehearsal",
                    "mode: tool",
                    "harness_build_slice",
                    "READ_ONLY",
                    "would be allowed by default",
                    "Next command",
                    "execution governor: continue building Jarvis V2 as an agent harness",
                ],
            ),
        ]
        for case, required in cases:
            approvals_before = len(runtime.store.list_pending_approvals(limit=100))
            result = runtime.handle(case)
            print(f"[ok={result.verified}] {case}")
            print(result.response[:2200])
            print()
            if not result.verified:
                raise SystemExit(f"Expected '{case}' to run as read-only.")
            missing = [item for item in required if item not in result.response]
            if missing:
                raise SystemExit(f"Assistant turn rehearsal missing expected text for '{case}': {missing}")
            metadata = result.tool_results[0].metadata
            _assert_rehearsal_handoff(
                metadata,
                key="assistant_turn_rehearsal_handoff",
                source="assistant_turn_rehearsal",
                include_context=True,
            )
            if (
                metadata.get("calls_model")
                or metadata.get("calls_chat_model")
                or metadata.get("executes_tools")
                or metadata.get("queues_approval")
                or metadata.get("writes_memory")
                or metadata.get("writes_files")
                or metadata.get("writes_notes")
                or metadata.get("controls_computer")
                or metadata.get("approves_request")
                or metadata.get("dismisses_request")
                or metadata.get("reads_private_data")
                or metadata.get("reads_personal_data")
                or metadata.get("executes_side_effect")
                or metadata.get("external_side_effect")
                or metadata.get("requires_approval")
                or metadata.get("speaks")
            ):
                raise SystemExit("Assistant turn rehearsal must not call models, execute tools, write memory, control the computer, or queue approvals.")
            if len(runtime.store.list_pending_approvals(limit=100)) != approvals_before:
                raise SystemExit("Assistant turn rehearsal should not queue pending approvals.")
            if case.startswith("assistant turn rehearsal: can"):
                if metadata.get("route") != "chat" or metadata.get("recommendation") != "ANSWER_IN_CHAT":
                    raise SystemExit(f"Chat rehearsal missed route metadata: {metadata}")
                if metadata.get("safe_to_execute_now") is not True:
                    raise SystemExit(f"Chat rehearsal should be safe to send: {metadata}")
            elif "run command" in case:
                if metadata.get("route") != "approval" or metadata.get("recommendation") != "ENTER_EXECUTION_GOVERNOR":
                    raise SystemExit(f"Risky tool rehearsal missed approval route metadata: {metadata}")
                if metadata.get("safe_to_execute_now") is not False:
                    raise SystemExit(f"Risky tool rehearsal should not be safe to execute now: {metadata}")
                if not str(metadata.get("next_command", "")).startswith("execution governor: "):
                    raise SystemExit(f"Risky tool rehearsal should route through execution governor first: {metadata}")
                commands = metadata.get("recommended_next_commands", [])
                if not commands or not str(commands[0]).startswith("execution governor: "):
                    raise SystemExit(f"Risky tool rehearsal recommended commands should begin with execution governor: {metadata}")
                for command in ["approval readiness latest", "approval packet latest", "approval chain proof latest"]:
                    if command not in metadata.get("recommended_next_commands", []):
                        raise SystemExit(f"Risky tool rehearsal missed approval-chain next command {command}: {metadata}")
            else:
                if metadata.get("route") != "auto_tool" or metadata.get("recommendation") != "AUTO_RUN_LOCAL_SAFE":
                    raise SystemExit(f"Local-safe rehearsal missed auto-tool route metadata: {metadata}")
                if metadata.get("safe_to_execute_now") is not True:
                    raise SystemExit(f"Local-safe rehearsal should be safe to execute now: {metadata}")
                if not str(metadata.get("next_command", "")).startswith("execution governor: "):
                    raise SystemExit(f"Local-safe rehearsal missed next command: {metadata}")

        memory_result = runtime.handle("search memory for previews should not write memory")
        if "No memories found" not in memory_result.response:
            raise SystemExit("Assistant turn rehearsal should not create memory while previewing remember.")

        bounded = runtime.registry.get("assistant_turn_rehearsal").handler({"message": "run command " + ("python3 --version " * 80)})
        print("[ok] direct bounded assistant_turn_rehearsal")
        print(bounded.output[:900])
        print()
        if not bounded.ok or len(bounded.metadata.get("message", "")) > 500:
            raise SystemExit(f"Assistant turn rehearsal message was not bounded: {bounded.metadata}")

        path_request = "run command cat /\x55sers/example/private.txt /private/tmp/token.txt /var/folders/zc/cache.txt /tmp/raw.txt"
        for tool_name, args in [
            ("assistant_turn_rehearsal", {"message": path_request}),
            ("action_rehearsal", {"request": path_request}),
            ("command_diagnosis", {"request": path_request}),
        ]:
            path_preview = runtime.registry.get(tool_name).handler(args)
            print(f"[ok] path-redacted {tool_name}")
            print(path_preview.output[:1200])
            print()
            combined = f"{path_preview.output}\n{path_preview.metadata}"
            for forbidden in ["/\x55sers/", "/private/", "/var/folders/", "/tmp/"]:
                if forbidden in combined:
                    raise SystemExit(f"{tool_name} leaked local path fragment {forbidden!r}: {combined}")
            if "<local-path>" not in combined:
                raise SystemExit(f"{tool_name} should retain a redacted local-path marker for operator review: {combined}")
            if path_preview.metadata.get("route") != "approval":
                raise SystemExit(f"{tool_name} should preserve internal high-risk routing despite redaction: {path_preview.metadata}")
            if tool_name == "command_diagnosis":
                _assert_command_diagnosis_handoff(path_preview.metadata)
            else:
                _assert_rehearsal_handoff(
                    path_preview.metadata,
                    key="assistant_turn_rehearsal_handoff" if tool_name == "assistant_turn_rehearsal" else "action_rehearsal_handoff",
                    source=tool_name,
                    include_context=tool_name == "assistant_turn_rehearsal",
                )
            if path_preview.metadata.get("safe_to_execute_now") is not False:
                raise SystemExit(f"{tool_name} should keep path-shaped shell preview unsafe to execute: {path_preview.metadata}")
            if path_preview.metadata.get("next_command") != "execution governor: run command cat <local-path>":
                raise SystemExit(f"{tool_name} should redact the next command handoff: {path_preview.metadata}")
            planned_actions = path_preview.metadata.get("planned_actions", [])
            if not planned_actions or planned_actions[0].get("tool") != "run_shell_command":
                raise SystemExit(f"{tool_name} should still route to run_shell_command internally: {path_preview.metadata}")
            planned_args = planned_actions[0].get("args", {})
            if planned_args.get("command") != "cat <local-path>":
                raise SystemExit(f"{tool_name} should redact planned command args: {path_preview.metadata}")

    with TemporaryDirectory(prefix="jarvis-turn-recovery-closure-") as temp:
        runtime = make_temp_runtime(Path(temp))
        run_id = runtime.store.log_tool_run(
            runtime.session_id,
            "checkpoint_recovery_execute",
            "LOCAL_SAFE",
            False,
            False,
            "reviewed local-safe step failed verification",
            metadata={"failure_kind": "verification_failed"},
        )
        cases = [
            ("assistant_turn_rehearsal", {"message": "remember that previews should close recovery first"}),
            ("action_rehearsal", {"request": "remember that previews should close recovery first"}),
            ("command_diagnosis", {"request": "remember that previews should close recovery first"}),
        ]
        for tool_name, args in cases:
            held = runtime.registry.get(tool_name).handler(args)
            if tool_name in {"assistant_turn_rehearsal", "action_rehearsal"}:
                _assert_rehearsal_handoff(
                    held.metadata,
                    key="assistant_turn_rehearsal_handoff" if tool_name == "assistant_turn_rehearsal" else "action_rehearsal_handoff",
                    source=tool_name,
                    include_context=tool_name == "assistant_turn_rehearsal",
                )
            elif tool_name == "command_diagnosis":
                _assert_command_diagnosis_handoff(held.metadata)
            print(f"[ok] recovery-held {tool_name}")
            print(held.output[:1100])
            print()
            if held.metadata.get("route") != "recovery_closure" or held.metadata.get("recommendation") != "RECOVERY_CLOSURE_REQUIRED":
                raise SystemExit(f"{tool_name} should hold local-safe action behind recovery closure: {held.metadata}")
            if held.metadata.get("safe_to_execute_now") is not False:
                raise SystemExit(f"{tool_name} missed recovery safe-to-execute blocker: {held.metadata}")
            for expected in [f"verification receipt {run_id}", f"execution recovery packet {run_id}", f"execution learning closure {run_id}", f"after-action learning packet {run_id}"]:
                if expected not in held.metadata.get("recovery_closure_required_commands", []):
                    raise SystemExit(f"{tool_name} missed required closure command {expected!r}: {held.metadata}")
            closure_commands = held.metadata.get("recovery_closure_required_commands", [])
            if held.metadata.get("recovery_closure_next_required_command") != closure_commands[0]:
                raise SystemExit(f"{tool_name} missed first recovery proof command: {held.metadata}")
            if held.metadata.get("recovery_closure_proof_queue") != closure_commands:
                raise SystemExit(f"{tool_name} recovery proof queue diverged from required commands: {held.metadata}")
            if held.metadata.get("recovery_closure_proof_queue_count") != len(closure_commands):
                raise SystemExit(f"{tool_name} recovery proof queue count diverged: {held.metadata}")
            if held.metadata.get("recovery_closure_next_proof_command") != closure_commands[0]:
                raise SystemExit(f"{tool_name} missed next recovery proof command alias: {held.metadata}")
            if "next required:" not in held.output:
                raise SystemExit(f"{tool_name} did not render the next required recovery command.")
            if "next required proof" in held.output:
                raise SystemExit(f"{tool_name} should not use ambiguous next required proof prose.")
            if "Execution health recovery closure:" not in held.output:
                raise SystemExit(f"{tool_name} did not render recovery closure details.")
            if "proof queue" not in held.output:
                raise SystemExit(f"{tool_name} did not render the recovery proof queue.")
            if held.metadata.get("execution_learning_state") != "LEARNING_DEBT_AFTER_FAILURE":
                raise SystemExit(f"{tool_name} missed failed-run execution learning debt state: {held.metadata}")
            learning_commands = held.metadata.get("execution_learning_required_commands", [])
            learning_closure_command = f"execution learning closure {run_id}"
            after_action_command = f"after-action learning packet {run_id}"
            if learning_closure_command not in learning_commands:
                raise SystemExit(f"{tool_name} missed failed-run execution learning closure command: {held.metadata}")
            if after_action_command not in learning_commands:
                raise SystemExit(f"{tool_name} missed failed-run execution learning command: {held.metadata}")
            if learning_commands.index(after_action_command) > learning_commands.index(learning_closure_command):
                raise SystemExit(f"{tool_name} should place after-action learning before execution learning closure: {held.metadata}")
            if held.metadata.get("execution_learning_proof_queue") != learning_commands:
                raise SystemExit(f"{tool_name} execution learning proof queue diverged: {held.metadata}")
            if held.metadata.get("execution_learning_proof_queue_count") != len(learning_commands):
                raise SystemExit(f"{tool_name} execution learning proof queue count diverged: {held.metadata}")
            if held.metadata.get("execution_learning_next_proof_command") != held.metadata.get("execution_learning_next_required_command"):
                raise SystemExit(f"{tool_name} execution learning next proof alias diverged: {held.metadata}")
            if "Execution learning debt:" not in held.output:
                raise SystemExit(f"{tool_name} did not render failed-run execution learning debt.")
            if "next learning proof:" in held.output:
                raise SystemExit(f"{tool_name} should not use ambiguous next learning proof prose.")

        chat_held = runtime.registry.get("assistant_turn_rehearsal").handler({"message": "can we talk about Jarvis memory?"})
        _assert_rehearsal_handoff(
            chat_held.metadata,
            key="assistant_turn_rehearsal_handoff",
            source="assistant_turn_rehearsal",
            include_context=True,
        )
        if chat_held.metadata.get("route") != "chat" or chat_held.metadata.get("safe_to_execute_now") is not True:
            raise SystemExit(f"Recovery closure should not block pure chat rehearsal: {chat_held.metadata}")

    with TemporaryDirectory(prefix="jarvis-turn-learning-debt-") as temp:
        runtime = make_temp_runtime(Path(temp))
        run_id = runtime.store.log_tool_run(
            runtime.session_id,
            "remember",
            "LOCAL_SAFE",
            True,
            False,
            "Remembered a useful local fact.",
            metadata={"fact": "assistant turn learning debt fixture"},
        )
        cases = [
            ("assistant_turn_rehearsal", {"message": "remember that previews should close learning debt first"}),
            ("action_rehearsal", {"request": "remember that previews should close learning debt first"}),
        ]
        for tool_name, args in cases:
            held = runtime.registry.get(tool_name).handler(args)
            _assert_rehearsal_handoff(
                held.metadata,
                key="assistant_turn_rehearsal_handoff" if tool_name == "assistant_turn_rehearsal" else "action_rehearsal_handoff",
                source=tool_name,
                include_context=tool_name == "assistant_turn_rehearsal",
            )
            print(f"[ok] successful-action-learning-context {tool_name}")
            print(held.output[:1100])
            print()
            if held.metadata.get("route") != "auto_tool":
                raise SystemExit(f"{tool_name} should not hold successful local action context behind learning debt: {held.metadata}")
            if held.metadata.get("safe_to_execute_now") is not True:
                raise SystemExit(f"{tool_name} should keep successful local action context safe to execute: {held.metadata}")
            if held.metadata.get("execution_learning_state") != "LEARNING_REVIEW_REQUIRED":
                raise SystemExit(f"{tool_name} missed learning debt state: {held.metadata}")
            expected_closure = f"execution learning closure {run_id}"
            expected = f"after-action learning packet {run_id}"
            if not str(held.metadata.get("next_command", "")).startswith("execution governor: "):
                raise SystemExit(f"{tool_name} should keep normal routing for successful action context: {held.metadata}")
            learning_commands = held.metadata.get("execution_learning_required_commands", [])
            if expected_closure not in learning_commands:
                raise SystemExit(f"{tool_name} missed required learning closure command {expected_closure!r}: {held.metadata}")
            if expected not in learning_commands:
                raise SystemExit(f"{tool_name} missed required learning command {expected!r}: {held.metadata}")
            if learning_commands.index(expected) > learning_commands.index(expected_closure):
                raise SystemExit(f"{tool_name} should require after-action learning before learning closure: {held.metadata}")
            if held.metadata.get("execution_learning_proof_queue") != learning_commands:
                raise SystemExit(f"{tool_name} execution learning proof queue diverged: {held.metadata}")
            if held.metadata.get("execution_learning_proof_queue_count") != len(learning_commands):
                raise SystemExit(f"{tool_name} execution learning proof queue count diverged: {held.metadata}")
            if held.metadata.get("execution_learning_next_proof_command") != held.metadata.get("execution_learning_next_required_command"):
                raise SystemExit(f"{tool_name} execution learning next proof alias diverged: {held.metadata}")
            if "Execution learning debt:" not in held.output:
                raise SystemExit(f"{tool_name} should render successful action learning context in tool rehearsal.")

        chat_held = runtime.registry.get("assistant_turn_rehearsal").handler({"message": "can we talk about Jarvis memory?"})
        _assert_rehearsal_handoff(
            chat_held.metadata,
            key="assistant_turn_rehearsal_handoff",
            source="assistant_turn_rehearsal",
            include_context=True,
        )
        if chat_held.metadata.get("route") != "chat" or chat_held.metadata.get("safe_to_execute_now") is not True:
            raise SystemExit(f"Learning debt should not block pure chat rehearsal: {chat_held.metadata}")


if __name__ == "__main__":
    main()
