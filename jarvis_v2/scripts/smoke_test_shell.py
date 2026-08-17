from __future__ import annotations

import json
import os
import re
import selectors
import signal
import shlex
import subprocess
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from jarvis_v2.agent.types import RiskLevel
from jarvis_v2.scripts.test_runtime import handle_runtime_case, make_temp_runtime
from jarvis_v2.tools.shell import (
    MAX_COMMAND_CHARS,
    _ProcessOutcome,
    _run_bounded_process,
    _terminate_and_reap,
    run_shell_command,
)


def assert_no_local_path(value: object, label: str) -> None:
    text = str(value)
    for fragment in ("/\x55sers/", "/private/", "/var/folders/", "/tmp/"):
        if fragment in text:
            raise SystemExit(f"{label} leaked a local path: {text}")


def assert_shell_refusal_handoff(result: object, reason: str, label: str) -> None:
    metadata = result.metadata
    handoff = metadata.get("shell_refusal_handoff")
    if not metadata.get("shell_refusal_handoff_ready") or not isinstance(handoff, dict):
        raise SystemExit(f"{label} should expose a shell refusal handoff: {metadata}")
    if handoff.get("source") != "run_shell_command" or handoff.get("mutation") != "shell_command_execute":
        raise SystemExit(f"{label} has the wrong shell refusal source/mutation: {handoff}")
    if handoff.get("reason") != reason or metadata.get("refusal_reason") != reason:
        raise SystemExit(f"{label} should keep refusal reason parity: {metadata}")
    if handoff.get("changed") != [] or not handoff.get("refused"):
        raise SystemExit(f"{label} should report refused/no-change state: {handoff}")
    if not handoff.get("approval_gated") or not handoff.get("requires_approval"):
        raise SystemExit(f"{label} should preserve approval-gated recovery metadata: {handoff}")
    boundaries = handoff.get("boundaries")
    if not isinstance(boundaries, dict):
        raise SystemExit(f"{label} should include refusal boundaries: {handoff}")
    false_flags = (
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
    )
    for flag in false_flags:
        if metadata.get(flag) or boundaries.get(flag):
            raise SystemExit(f"{label} should keep {flag}=False on refusal: {metadata}")
    if not any(str(command).startswith("run command") for command in handoff.get("next_commands", [])):
        raise SystemExit(f"{label} should include a command-first recovery hint: {handoff}")
    assert_no_local_path(handoff, f"{label} handoff")


def assert_conservative_execution_boundaries(result: object, label: str) -> None:
    metadata = result.metadata
    for flag in (
        "reads_private_data",
        "reads_personal_data",
        "writes_files",
        "executes_side_effect",
        "executes_tools",
        "requires_approval",
        "approval_gated",
    ):
        if metadata.get(flag) is not True:
            raise SystemExit(f"{label} should conservatively report {flag}=True: {metadata}")


def mocked_process_outcome(
    *,
    returncode: int | None = 0,
    stdout: bytes = b"",
    stderr: bytes = b"",
    stdout_bytes_seen: int | None = None,
    stderr_bytes_seen: int | None = None,
    stdout_truncated: bool = False,
    stderr_truncated: bool = False,
    timed_out: bool = False,
    post_start_failed: bool = False,
    exception_type: str | None = None,
    termination_escalated: bool = False,
    process_reaped: bool = True,
    capture_limit: int = 24000,
) -> _ProcessOutcome:
    return _ProcessOutcome(
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
        stdout_bytes_seen=len(stdout) if stdout_bytes_seen is None else stdout_bytes_seen,
        stderr_bytes_seen=len(stderr) if stderr_bytes_seen is None else stderr_bytes_seen,
        capture_limit_bytes_per_stream=capture_limit,
        stdout_truncated=stdout_truncated,
        stderr_truncated=stderr_truncated,
        timed_out=timed_out,
        post_start_failed=post_start_failed,
        exception_type=exception_type,
        termination_escalated=termination_escalated,
        process_reaped=process_reaped,
    )


def main() -> None:
    with TemporaryDirectory(prefix="jarvis-shell-") as temp:
        runtime = make_temp_runtime(Path(temp))
        if runtime.registry.get("run_shell_command").risk != RiskLevel.HIGH_RISK:
            raise SystemExit("run_shell_command must remain HIGH_RISK.")
        missing_command_phrases = [
            "run command",
            "run command please",
            "run a shell command",
            "run shell command",
            "execute shell command",
            "terminal command",
            "run terminal command",
        ]
        for phrase in missing_command_phrases:
            approvals_before = len(runtime.store.list_pending_approvals(limit=100))
            result = runtime.handle(phrase)
            approvals_after = len(runtime.store.list_pending_approvals(limit=100))
            if approvals_after != approvals_before:
                raise SystemExit(f"{phrase!r} should not queue an approval without a concrete command")
            if result.tool_results:
                raise SystemExit(f"{phrase!r} should not execute tool_detail or shell tools: {result.tool_results!r}")
            response = result.response
            expected_fragments = ["run_shell_command", "risk level HIGH_RISK", "concrete command", "run command <command>"]
            for fragment in expected_fragments:
                if fragment not in response:
                    raise SystemExit(f"{phrase!r} missed shell stub recovery fragment {fragment!r}: {response!r}")
            chat_response = (result.metadata or {}).get("chat_response") or {}
            if chat_response.get("runtime_route") != "command_suggestion":
                raise SystemExit(f"{phrase!r} should stay on the command-suggestion route: {chat_response!r}")

        cases = [
            ("list tools code", False),
            ("run command python3 --version", False),
            ("run command python3 --version", True),
            ("run command python3 --version && echo nope", True),
        ]
        for case, approved in cases:
            result = handle_runtime_case(runtime, case, approved=approved)
            status = "ok" if result.verified else "blocked"
            approval = " approved" if approved else ""
            print(f"[{status}{approval}] {case}")
            print(result.response[:1200])
            print()
            if approved and result.tool_results:
                metadata = result.tool_results[0].metadata
                if not metadata.get("approval_gated") or not metadata.get("requires_approval"):
                    raise SystemExit("Shell command metadata should keep the approval boundary visible.")
                if metadata.get("calls_model") or metadata.get("controls_computer") or metadata.get("queues_approval"):
                    raise SystemExit(f"Shell command metadata has unsafe flags: {metadata}")
                if "&&" in case and not metadata.get("blocked"):
                    raise SystemExit("Shell control operator rejection should be marked as blocked.")
                if case == "run command python3 --version" and approved:
                    for key in ("executed_handler", "execution_started", "action_attempted", "process_reaped"):
                        if metadata.get(key) is not True:
                            raise SystemExit(f"Approved shell run missed {key} execution proof: {metadata}")
                    run_id = metadata.get("logged_tool_run_id")
                    stored = runtime.store.get_tool_run(run_id) if isinstance(run_id, int) else None
                    if stored is None:
                        raise SystemExit(f"Approved shell run missed durable audit row: {metadata}")
                    stored_metadata = json.loads(stored["metadata"] or "{}")
                    for key in (
                        "executed_handler",
                        "execution_started",
                        "action_attempted",
                        "side_effect_possible",
                        "output_capture_streaming",
                        "process_reaped",
                    ):
                        if stored_metadata.get(key) is not True:
                            raise SystemExit(f"Approved shell audit row missed {key}: {stored_metadata}")
                    runtime_trace = (result.metadata or {}).get("runtime_trace") or {}
                    traced_results = runtime_trace.get("tool_results") or []
                    traced_shell = next(
                        (item for item in traced_results if item.get("tool_name") == "run_shell_command"),
                        None,
                    )
                    if not isinstance(traced_shell, dict):
                        raise SystemExit(f"Approved shell run missed runtime trace result: {runtime_trace}")
                    for key in (
                        "executed_handler",
                        "execution_started",
                        "action_attempted",
                        "side_effect_possible",
                        "output_capture_streaming",
                        "process_reaped",
                    ):
                        if traced_shell.get(key) is not True:
                            raise SystemExit(f"Approved shell runtime trace missed {key}: {traced_shell}")
                    if traced_shell.get("capture_limit_bytes_per_stream") != 24000:
                        raise SystemExit(f"Approved shell runtime trace missed its capture cap: {traced_shell}")
                    if traced_shell.get("stdout_bytes_seen", 0) < 1 or traced_shell.get("stderr_bytes_seen") != 0:
                        raise SystemExit(f"Approved shell runtime trace missed observed-byte counts: {traced_shell}")

        long_result = handle_runtime_case(runtime, "run command " + ("python3 " * 200), approved=True)
        if long_result.tool_results:
            metadata = long_result.tool_results[0].metadata
            command_text = metadata.get("command", "")
            if isinstance(command_text, str) and len(command_text) > 620:
                raise SystemExit("Shell command text was not bounded.")
            if not metadata.get("approval_gated"):
                raise SystemExit("Long shell command should retain approval-gated metadata.")

        oversized_command = "python3 " + ("x" * MAX_COMMAND_CHARS)
        with patch("jarvis_v2.tools.shell._run_bounded_process") as run_mock:
            oversized = run_shell_command({"command": oversized_command})
        if run_mock.called:
            raise SystemExit("Oversized shell command must be rejected before spawning.")
        expected_oversized_output = f"Command is too long. Maximum length is {MAX_COMMAND_CHARS} characters."
        if oversized.ok or oversized.output != expected_oversized_output:
            raise SystemExit(f"Oversized shell command returned the wrong diagnostic: {oversized.output!r}")
        if not oversized.metadata.get("command_too_long") or oversized.metadata.get("command_chars") != len(oversized_command):
            raise SystemExit(f"Oversized shell command lost length diagnostics: {oversized.metadata}")
        assert_shell_refusal_handoff(oversized, "command_too_long", "oversized shell command")

        exact_command = "python3 -c 'print(\"  exact spacing  \" )' --dry-run"
        with patch(
            "jarvis_v2.tools.shell._run_bounded_process",
            return_value=mocked_process_outcome(stdout=b"ok\n"),
        ) as run_mock:
            exact = run_shell_command({"command": exact_command})
        expected_argv = ["python3", "-c", 'print("  exact spacing  " )', "--dry-run"]
        if not exact.ok or exact.metadata.get("command") != "python3 <arguments-redacted>":
            raise SystemExit(f"Normal shell receipt should withhold command arguments: {exact.metadata}")
        if run_mock.call_args.args[0] != expected_argv or run_mock.call_args.kwargs.get("cwd") != Path("."):
            raise SystemExit(f"Normal shell command spawned with changed arguments: {run_mock.call_args}")
        if run_mock.call_args.kwargs.get("timeout") != 30 or run_mock.call_args.kwargs.get("max_chars") != 6000:
            raise SystemExit(f"Normal shell command lost execution bounds: {run_mock.call_args}")
        if 'print("  exact spacing  " )' in str(exact.metadata):
            raise SystemExit(f"Normal shell receipt leaked command payload: {exact.metadata}")
        if exact.metadata.get("output_capture_streaming") is not True or exact.metadata.get("process_reaped") is not True:
            raise SystemExit(f"Normal shell receipt missed streaming/reaping proof: {exact.metadata}")
        assert_conservative_execution_boundaries(exact, "normal shell command")

        real_popen = subprocess.Popen
        with patch("jarvis_v2.tools.shell.subprocess.Popen", wraps=real_popen) as popen_mock:
            configured_spawn = run_shell_command({"command": "python3 --version", "max_chars": 500})
        if not configured_spawn.ok:
            raise SystemExit(f"Configured streaming spawn failed: {configured_spawn}")
        popen_kwargs = popen_mock.call_args.kwargs
        for key, expected in {
            "stdout": subprocess.PIPE,
            "stderr": subprocess.PIPE,
            "text": False,
            "bufsize": 0,
            "start_new_session": True,
            "shell": False,
        }.items():
            if popen_kwargs.get(key) != expected:
                raise SystemExit(f"Streaming spawn missed {key}={expected!r}: {popen_mock.call_args}")

        literal_args_code = 'import sys\nprint(",".join(sys.argv[1:]))'
        literal_args_command = (
            f"python3 -c {shlex.quote(literal_args_code)} "
            f"{shlex.quote('a b')} {shlex.quote('*.txt')} {shlex.quote('$HOME')}"
        )
        literal_args = run_shell_command({"command": literal_args_command, "max_chars": 500})
        if not literal_args.ok or literal_args.output != "a b,*.txt,$HOME":
            raise SystemExit(f"Direct argv gained shell expansion or lost spacing: {literal_args}")

        cat_command = "cat relative-private-payload.txt"
        cat_cwd = "/private/tmp/jarvis-shell-private-cwd"
        cat_stdout = ("read /\x55sers/example/private-file.txt\n" + ("x" * 300) + "\n").encode()
        cat_stderr = ("warning /tmp/jarvis-shell-stderr.txt\n" + ("y" * 300)).encode()
        cat_completed = mocked_process_outcome(
            stdout=cat_stdout,
            stderr=cat_stderr,
        )
        with patch("jarvis_v2.tools.shell._run_bounded_process", return_value=cat_completed) as run_mock:
            cat_result = run_shell_command(
                {"command": cat_command, "cwd": cat_cwd, "max_chars": 500}
            )
        if run_mock.call_args.args[0] != ["cat", "relative-private-payload.txt"]:
            raise SystemExit(f"Mocked cat command did not receive its exact argv: {run_mock.call_args}")
        if run_mock.call_args.kwargs.get("cwd") != Path(cat_cwd):
            raise SystemExit(f"Mocked cat command did not receive its exact cwd: {run_mock.call_args}")
        if cat_result.metadata.get("command") != "cat <arguments-redacted>" or cat_result.metadata.get("cwd") != "<local-path>":
            raise SystemExit(f"Mocked cat receipt did not redact command/cwd metadata: {cat_result.metadata}")
        if "relative-private-payload.txt" in str(cat_result.metadata):
            raise SystemExit(f"Mocked cat receipt leaked command payload: {cat_result.metadata}")
        if len(cat_result.output) > 500 or "... truncated" not in cat_result.output or "stderr:" not in cat_result.output:
            raise SystemExit(f"Mocked cat output was not bounded: {cat_result.output!r}")
        assert_no_local_path(cat_result.output, "mocked cat stdout/stderr")
        assert_no_local_path(cat_result.metadata, "mocked cat metadata")
        assert_conservative_execution_boundaries(cat_result, "mocked cat command")

        rm_command = "rm relative-delete-payload.txt"
        rm_completed = mocked_process_outcome()
        with patch("jarvis_v2.tools.shell._run_bounded_process", return_value=rm_completed) as run_mock:
            rm_result = run_shell_command({"command": rm_command})
        if run_mock.call_args.args[0] != ["rm", "relative-delete-payload.txt"]:
            raise SystemExit(f"Mocked rm command did not receive its exact argv: {run_mock.call_args}")
        if not rm_result.ok or rm_result.metadata.get("command") != "rm <arguments-redacted>":
            raise SystemExit(f"Mocked rm receipt should succeed without exposing arguments: {rm_result.metadata}")
        if "relative-delete-payload.txt" in str(rm_result.metadata):
            raise SystemExit(f"Mocked rm receipt leaked command payload: {rm_result.metadata}")
        assert_conservative_execution_boundaries(rm_result, "mocked rm command")

        timeout_outcome = mocked_process_outcome(
            returncode=-15,
            stdout=(b"started /tmp/jarvis-timeout\n" + (b"x" * 500)),
            stderr=b"stderr /\x55sers/example/private-value\n",
            stdout_bytes_seen=930,
            stdout_truncated=True,
            timed_out=True,
            capture_limit=500,
        )
        with patch("jarvis_v2.tools.shell._run_bounded_process", return_value=timeout_outcome):
            timed_out = run_shell_command({"command": "python3 --version", "timeout": 7, "max_chars": 500})
        if timed_out.ok or not timed_out.metadata.get("timed_out"):
            raise SystemExit(f"Timed-out shell command should remain failed: {timed_out.metadata}")
        if timed_out.metadata.get("execution_started") is not True or timed_out.metadata.get("side_effect_possible") is not True:
            raise SystemExit(f"Timed-out shell command should report possible effects: {timed_out.metadata}")
        if not timed_out.metadata.get("executes_side_effect") or timed_out.metadata.get("start_failed"):
            raise SystemExit(f"Timed-out shell command must not claim safe non-execution: {timed_out.metadata}")
        if "Execution started and may have produced side effects" not in timed_out.output or len(timed_out.output) > 500:
            raise SystemExit(f"Timed-out shell command should return a bounded truthful diagnostic: {timed_out.output!r}")
        assert_no_local_path(timed_out.output, "timed-out shell command output")
        assert_conservative_execution_boundaries(timed_out, "timed-out shell command")

        timeout_runtime = make_temp_runtime(Path(temp) / "timeout-approval")
        timeout_request = "run command python3 --version"
        held_timeout = timeout_runtime.handle(timeout_request)
        timeout_approval_id = held_timeout.tool_results[0].metadata.get("approval_id")
        if not isinstance(timeout_approval_id, int):
            raise SystemExit(f"Timeout approval fixture did not queue exactly: {held_timeout.tool_results}")
        timeout_readiness = timeout_runtime.registry.get("approval_readiness_packet").handler(
            {"approval_id": timeout_approval_id}
        )
        timeout_packet = timeout_runtime.registry.get("approval_execution_packet").handler(
            {"approval_id": timeout_approval_id}
        )
        timeout_transition = timeout_runtime.registry.get("approve_pending_approval").handler(
            {"approval_id": timeout_approval_id}
        )
        if not timeout_readiness.ok or not timeout_packet.ok or not timeout_transition.ok:
            raise SystemExit("Timeout approval fixture could not complete its required last-look transition.")
        with patch("jarvis_v2.tools.shell._run_bounded_process", return_value=timeout_outcome):
            approved_timeout = timeout_runtime.handle(
                timeout_request,
                approved=True,
                approved_approval_id=timeout_approval_id,
            )
        if approved_timeout.verified or len(approved_timeout.tool_results) != 1:
            raise SystemExit(f"Approved shell timeout should remain an unverified single attempt: {approved_timeout}")
        approved_timeout_result = approved_timeout.tool_results[0]
        approved_timeout_metadata = approved_timeout_result.metadata
        for key in (
            "executed_handler",
            "execution_started",
            "action_attempted",
            "side_effect_possible",
            "output_capture_streaming",
            "timed_out",
            "process_reaped",
        ):
            if approved_timeout_metadata.get(key) is not True:
                raise SystemExit(f"Approved shell timeout missed {key}: {approved_timeout_metadata}")
        timeout_run_id = approved_timeout_metadata.get("logged_tool_run_id")
        timeout_row = timeout_runtime.store.get_tool_run(timeout_run_id) if isinstance(timeout_run_id, int) else None
        if timeout_row is None or timeout_row["ok"] or not timeout_row["approved"]:
            raise SystemExit(f"Approved shell timeout missed its failed approved audit row: {timeout_row}")
        timeout_stored_metadata = json.loads(timeout_row["metadata"] or "{}")
        for key in (
            "executed_handler",
            "execution_started",
            "action_attempted",
            "side_effect_possible",
            "output_capture_streaming",
            "timed_out",
            "process_reaped",
        ):
            if timeout_stored_metadata.get(key) is not True:
                raise SystemExit(f"Approved timeout audit row missed {key}: {timeout_stored_metadata}")
        timeout_claim = timeout_runtime.store.get_approval_execution_claim(timeout_approval_id)
        if timeout_claim is None or timeout_claim["outcome"] != "outcome_unknown" or not timeout_claim["completed_at"]:
            raise SystemExit(f"Approved shell timeout did not finalize one-shot uncertainty: {timeout_claim}")
        timeout_evidence = timeout_runtime.store.classify_approval_execution_evidence(timeout_approval_id)
        if not timeout_evidence.outcome_unknown or timeout_evidence.valid_execution_proof:
            raise SystemExit(f"Approved shell timeout overclaimed durable outcome truth: {timeout_evidence}")
        timeout_trace_receipt = timeout_runtime.handle("runtime trace receipt")
        for expected in (
            "execution started: yes",
            "direct process reaped: yes",
            "output capture: bounded streaming",
            "cleanup: original process group stop attempted",
            "independently detached descendants are not verified",
        ):
            if expected not in timeout_trace_receipt.response:
                raise SystemExit(f"Approved timeout trace receipt missed {expected!r}: {timeout_trace_receipt.response}")
        with patch("jarvis_v2.tools.shell._run_bounded_process") as replay_runner:
            timeout_replay = timeout_runtime.handle(
                timeout_request,
                approved=True,
                approved_approval_id=timeout_approval_id,
            )
        if replay_runner.called or "already been used" not in timeout_replay.response:
            raise SystemExit(f"Failed approved shell attempt was replayable: {timeout_replay}")

        invalid_utf8 = mocked_process_outcome(stdout=b"prefix \xff suffix\n")
        with patch("jarvis_v2.tools.shell._run_bounded_process", return_value=invalid_utf8):
            decoded = run_shell_command({"command": "python3 --version"})
        if not decoded.ok or "prefix \ufffd suffix" not in decoded.output:
            raise SystemExit(f"Invalid UTF-8 should be safely replaced in a completed receipt: {decoded}")
        if decoded.metadata.get("decode_failed") or decoded.metadata.get("output_capture_failed"):
            raise SystemExit(f"Replacement decoding must not invent an execution failure: {decoded.metadata}")
        assert_conservative_execution_boundaries(decoded, "replacement-decoded shell output")

        capture_failure = mocked_process_outcome(
            returncode=-15,
            stdout=b"partial output\n",
            post_start_failed=True,
            exception_type="OSError",
        )
        with patch("jarvis_v2.tools.shell._run_bounded_process", return_value=capture_failure):
            capture_failed = run_shell_command({"command": "python3 --version"})
        if capture_failed.ok or not capture_failed.metadata.get("output_capture_failed"):
            raise SystemExit(f"Post-start capture failure should remain failed: {capture_failed.metadata}")
        if capture_failed.metadata.get("execution_started") is not True or not capture_failed.metadata.get("process_reaped"):
            raise SystemExit(f"Post-start capture failure should report start/reaping truth: {capture_failed.metadata}")
        if "may have produced side effects" not in capture_failed.output or "partial output" not in capture_failed.output:
            raise SystemExit(f"Post-start capture failure lost truthful bounded output: {capture_failed.output!r}")
        assert_conservative_execution_boundaries(capture_failed, "post-start capture failure")

        read_failure_started = time.monotonic()
        with patch(
            "jarvis_v2.tools.shell._read_ready_streams",
            return_value=OSError("mocked pipe read failure"),
        ):
            read_failure_outcome = _run_bounded_process(
                ["python3", "-c", "import time\ntime.sleep(5)"],
                cwd=Path("."),
                timeout=5,
                max_chars=500,
            )
        if time.monotonic() - read_failure_started > 2.0:
            raise SystemExit("Post-spawn read failure cleanup exceeded its bounded grace period.")
        if not read_failure_outcome.post_start_failed or read_failure_outcome.exception_type != "OSError":
            raise SystemExit(f"Real post-spawn read failure lost its classification: {read_failure_outcome}")
        if not read_failure_outcome.process_reaped or read_failure_outcome.timed_out:
            raise SystemExit(f"Post-spawn read failure did not terminate/reap promptly: {read_failure_outcome}")

        selector_failure_started = time.monotonic()
        with patch(
            "jarvis_v2.tools.shell.selectors.DefaultSelector",
            side_effect=OSError("mocked selector construction failure"),
        ):
            selector_failure_outcome = _run_bounded_process(
                ["python3", "-c", "import time\ntime.sleep(5)"],
                cwd=Path("."),
                timeout=5,
                max_chars=500,
            )
        if time.monotonic() - selector_failure_started > 2.0:
            raise SystemExit("Selector-construction cleanup exceeded its bounded grace period.")
        if not selector_failure_outcome.post_start_failed or selector_failure_outcome.streaming_started:
            raise SystemExit(f"Selector setup failure claimed streaming began: {selector_failure_outcome}")
        if selector_failure_outcome.exception_type != "OSError" or not selector_failure_outcome.process_reaped:
            raise SystemExit(f"Selector setup failure did not terminate/reap the spawned child: {selector_failure_outcome}")

        captured_processes: list[subprocess.Popen[bytes]] = []
        real_popen_for_failure = subprocess.Popen

        def capture_popen(*args: object, **kwargs: object) -> subprocess.Popen[bytes]:
            process = real_popen_for_failure(*args, **kwargs)  # type: ignore[arg-type]
            captured_processes.append(process)
            return process

        with patch("jarvis_v2.tools.shell.subprocess.Popen", side_effect=capture_popen), patch(
            "jarvis_v2.tools.shell.os.set_blocking",
            side_effect=OSError("mocked nonblocking setup failure"),
        ):
            set_blocking_failure = _run_bounded_process(
                ["python3", "-c", "import time\ntime.sleep(5)"],
                cwd=Path("."),
                timeout=5,
                max_chars=500,
            )
        set_blocking_process = captured_processes.pop()
        if not set_blocking_failure.post_start_failed or not set_blocking_failure.process_reaped:
            raise SystemExit(f"Nonblocking setup failure did not clean up execution: {set_blocking_failure}")
        if not set_blocking_process.stdout.closed or not set_blocking_process.stderr.closed:
            raise SystemExit("Nonblocking setup failure leaked a subprocess pipe descriptor.")

        class FailSecondRegisterSelector:
            def __init__(self) -> None:
                self.inner = selectors.DefaultSelector()
                self.register_calls = 0

            def register(self, fileobj: object, events: int, data: object = None) -> object:
                self.register_calls += 1
                if self.register_calls == 2:
                    raise OSError("mocked second-register failure")
                return self.inner.register(fileobj, events, data)

            def unregister(self, fileobj: object) -> object:
                return self.inner.unregister(fileobj)

            def select(self, timeout: float | None = None) -> list[tuple[object, int]]:
                return self.inner.select(timeout)  # type: ignore[return-value]

            def get_map(self) -> object:
                return self.inner.get_map()

            def close(self) -> None:
                self.inner.close()

        partial_selector = FailSecondRegisterSelector()
        with patch("jarvis_v2.tools.shell.subprocess.Popen", side_effect=capture_popen), patch(
            "jarvis_v2.tools.shell.selectors.DefaultSelector",
            return_value=partial_selector,
        ):
            partial_register_failure = _run_bounded_process(
                ["python3", "-c", "import time\ntime.sleep(5)"],
                cwd=Path("."),
                timeout=5,
                max_chars=500,
            )
        partial_register_process = captured_processes.pop()
        if not partial_register_failure.post_start_failed or not partial_register_failure.process_reaped:
            raise SystemExit(f"Partial selector registration did not clean up execution: {partial_register_failure}")
        if not partial_register_process.stdout.closed or not partial_register_process.stderr.closed:
            raise SystemExit("Partial selector registration leaked a subprocess pipe descriptor.")

        large_output_code = (
            "import os\n"
            "for _ in range(50):\n"
            " os.write(1,b'x'*4096)\n"
            " os.write(2,b'y'*4096)"
        )
        large_output_command = f"python3 -c {shlex.quote(large_output_code)}"
        with patch.object(
            subprocess.Popen,
            "communicate",
            side_effect=AssertionError("bounded runner must not call communicate"),
        ):
            large_outcome = _run_bounded_process(
                ["python3", "-c", large_output_code],
                cwd=Path("."),
                timeout=10,
                max_chars=500,
            )
        if len(large_outcome.stdout) > 2000 or len(large_outcome.stderr) > 2000:
            raise SystemExit(f"Streaming runner retained bytes beyond its fixed cap: {large_outcome}")
        if large_outcome.stdout_bytes_seen != 204800 or large_outcome.stderr_bytes_seen != 204800:
            raise SystemExit(f"Streaming runner did not drain both large pipes: {large_outcome}")
        with patch("jarvis_v2.tools.shell._run_bounded_process", return_value=large_outcome):
            large_output = run_shell_command(
                {"command": large_output_command, "timeout": 10, "max_chars": 500}
            )
        if not large_output.ok or len(large_output.output) > 500 or "... truncated" not in large_output.output:
            raise SystemExit(f"Real mixed large output was not bounded: {large_output}")
        large_metadata = large_output.metadata
        if large_metadata.get("stdout_bytes_seen") != 204800 or large_metadata.get("stderr_bytes_seen") != 204800:
            raise SystemExit(f"Streaming runner did not drain both large pipes: {large_metadata}")
        if not large_metadata.get("stdout_truncated") or not large_metadata.get("stderr_truncated"):
            raise SystemExit(f"Streaming runner missed per-pipe truncation: {large_metadata}")
        if large_metadata.get("capture_limit_bytes_per_stream") != 2000 or not large_metadata.get("process_reaped"):
            raise SystemExit(f"Streaming runner lost bounded-capture/reaping proof: {large_metadata}")
        assert_conservative_execution_boundaries(large_output, "real bounded large-output command")

        unicode_command = (
            "python3 -c 'import os\n"
            "value=\"한🙂\".encode()\n"
            "for byte in value:\n os.write(1,bytes([byte]))'"
        )
        with patch("jarvis_v2.tools.shell.STREAM_READ_CHUNK_BYTES", 1):
            unicode_output = run_shell_command({"command": unicode_command, "max_chars": 500})
        if not unicode_output.ok or unicode_output.output != "한🙂":
            raise SystemExit(f"Split UTF-8 output was not reconstructed exactly: {unicode_output}")
        if unicode_output.metadata.get("output_truncated") or not unicode_output.metadata.get("process_reaped"):
            raise SystemExit(f"Split UTF-8 output has incorrect lifecycle metadata: {unicode_output.metadata}")

        emoji_output = run_shell_command(
            {"command": "python3 -c 'print(\"🙂\"*501,end=\"\")'", "max_chars": 500}
        )
        if not emoji_output.ok or len(emoji_output.output) > 500 or "... truncated" not in emoji_output.output:
            raise SystemExit(f"Byte-truncated emoji output missed a visible marker: {emoji_output}")
        if not emoji_output.metadata.get("stdout_truncated") or not emoji_output.metadata.get("output_truncated"):
            raise SystemExit(f"Byte-truncated emoji output missed metadata truth: {emoji_output.metadata}")

        timeout_command = (
            "python3 -c 'import time\n"
            "print(\"started\",flush=True)\n"
            "time.sleep(5)'"
        )
        timeout_started = time.monotonic()
        real_timeout = run_shell_command(
            {"command": timeout_command, "timeout": 1, "max_chars": 500}
        )
        timeout_elapsed = time.monotonic() - timeout_started
        if real_timeout.ok or not real_timeout.metadata.get("timed_out"):
            raise SystemExit(f"Real timeout did not fail truthfully: {real_timeout}")
        if not real_timeout.metadata.get("process_reaped") or timeout_elapsed > 3.5:
            raise SystemExit(
                f"Timed-out process was not promptly reaped ({timeout_elapsed:.2f}s): {real_timeout.metadata}"
            )
        if "started" not in real_timeout.output or "may have produced side effects" not in real_timeout.output:
            raise SystemExit(f"Real timeout lost bounded partial output: {real_timeout.output!r}")
        assert_conservative_execution_boundaries(real_timeout, "real timed-out command")

        ignoring_child_code = (
            "import signal,time\n"
            "signal.signal(signal.SIGTERM,signal.SIG_IGN)\n"
            "time.sleep(30)"
        )
        process_tree_code = (
            "import os,subprocess\n"
            f"child=subprocess.Popen(['python3','-c',{ignoring_child_code!r}])\n"
            "print(f'TREE_PIDS={os.getpid()},{child.pid}',flush=True)"
        )
        process_tree_command = f"python3 -c {shlex.quote(process_tree_code)}"
        process_tree_timeout = run_shell_command(
            {"command": process_tree_command, "timeout": 1, "max_chars": 500}
        )
        match = re.search(r"TREE_PIDS=(\d+),(\d+)", process_tree_timeout.output)
        if not match:
            raise SystemExit(f"Process-tree timeout lost PID proof: {process_tree_timeout.output!r}")
        process_group_id = int(match.group(1))
        group_alive = True
        group_deadline = time.monotonic() + 1.0
        while time.monotonic() < group_deadline:
            try:
                os.killpg(process_group_id, 0)
            except ProcessLookupError:
                group_alive = False
                break
            time.sleep(0.02)
        if group_alive:
            try:
                os.killpg(process_group_id, signal.SIGKILL)
            except ProcessLookupError:
                group_alive = False
        if group_alive:
            raise SystemExit("TERM-ignoring descendant survived bounded process-group cleanup.")
        if not process_tree_timeout.metadata.get("timed_out") or not process_tree_timeout.metadata.get("termination_escalated"):
            raise SystemExit(f"Process-tree timeout missed SIGKILL escalation evidence: {process_tree_timeout.metadata}")
        if not process_tree_timeout.metadata.get("process_reaped"):
            raise SystemExit(f"Process-tree timeout missed direct-child reaping evidence: {process_tree_timeout.metadata}")
        if process_tree_timeout.metadata.get("all_descendants_terminated_verified") is not False:
            raise SystemExit(f"Process-tree timeout overclaimed detached-descendant verification: {process_tree_timeout.metadata}")
        if not process_tree_timeout.metadata.get("detached_descendants_may_survive"):
            raise SystemExit(f"Process-tree timeout hid the detached-descendant boundary: {process_tree_timeout.metadata}")

        class EscalatingProcess:
            pid = 43210

            def __init__(self) -> None:
                self.returncode: int | None = None
                self.wait_calls = 0

            def poll(self) -> int | None:
                return self.returncode

            def wait(self, timeout: float) -> int:
                self.wait_calls += 1
                if self.wait_calls == 1:
                    raise subprocess.TimeoutExpired(["mocked"], timeout)
                self.returncode = -signal.SIGKILL
                return self.returncode

        escalating_process = EscalatingProcess()
        with patch("jarvis_v2.tools.shell._signal_process_group") as signal_group, patch(
            "jarvis_v2.tools.shell._process_group_exists",
            return_value=True,
        ):
            escalated, reaped = _terminate_and_reap(escalating_process)  # type: ignore[arg-type]
        expected_signals = [
            ((escalating_process, signal.SIGTERM),),
            ((escalating_process, signal.SIGKILL),),
        ]
        if [call.args for call in signal_group.call_args_list] != [item[0] for item in expected_signals]:
            raise SystemExit(f"Timeout cleanup used the wrong process-group signals: {signal_group.call_args_list}")
        if not escalated or not reaped or escalating_process.wait_calls != 2:
            raise SystemExit("Timeout cleanup did not escalate and reap after the grace period.")

        class ExitedParentProcess:
            pid = 43211
            returncode = 0

            def poll(self) -> int:
                return self.returncode

        exited_parent_process = ExitedParentProcess()
        with patch("jarvis_v2.tools.shell._signal_process_group") as signal_group, patch(
            "jarvis_v2.tools.shell._process_group_exists",
            return_value=True,
        ), patch(
            "jarvis_v2.tools.shell.time.monotonic",
            side_effect=[100.0, 100.4, 100.6],
        ), patch("jarvis_v2.tools.shell.time.sleep") as sleep:
            escalated, reaped = _terminate_and_reap(exited_parent_process)  # type: ignore[arg-type]
        if not escalated or not reaped:
            raise SystemExit("Expired TERM grace should escalate after an exited parent.")
        sleep.assert_not_called()
        if [call.args[1] for call in signal_group.call_args_list] != [signal.SIGTERM, signal.SIGKILL]:
            raise SystemExit(f"Expired TERM grace used the wrong process-group signals: {signal_group.call_args_list}")

        nonzero = run_shell_command(
            {"command": "python3 -c 'raise SystemExit(7)'", "max_chars": 500}
        )
        if nonzero.ok or nonzero.metadata.get("returncode") != 7 or not nonzero.metadata.get("process_reaped"):
            raise SystemExit(f"Nonzero shell exit lost its code/reaping receipt: {nonzero}")

        malformed = run_shell_command({"command": "python3 'unterminated"})
        if malformed.ok or "Could not parse command. Check quoting" not in malformed.output:
            raise SystemExit(f"Malformed shell command should return stable parse guidance: {malformed.output}")
        if "No closing quotation" in malformed.output or malformed.metadata.get("exception_type") != "ValueError":
            raise SystemExit(f"Malformed shell command leaked raw parse text or lost diagnostics: {malformed.metadata}")
        if not malformed.metadata.get("approval_gated") or not malformed.metadata.get("parse_error"):
            raise SystemExit(f"Malformed shell command should retain approval parse metadata: {malformed.metadata}")
        assert_shell_refusal_handoff(malformed, "parse_error", "malformed shell command")

        missing = run_shell_command({"command": ""})
        if missing.ok or "No command provided" not in missing.output:
            raise SystemExit(f"Missing shell command should fail with stable guidance: {missing.output}")
        assert_shell_refusal_handoff(missing, "missing_command", "missing shell command")

        with patch("jarvis_v2.tools.shell._run_bounded_process") as run_mock:
            blocked_path = run_shell_command({"command": "echo /tmp/jarvis-shell-path && echo no"})
        if run_mock.called:
            raise SystemExit("Path-shaped blocked shell command must be rejected before spawning.")
        if blocked_path.ok or not blocked_path.metadata.get("blocked"):
            raise SystemExit(f"Path-shaped blocked shell command should still refuse operators: {blocked_path.metadata}")
        assert_shell_refusal_handoff(blocked_path, "blocked_control_operator", "blocked shell command")
        assert_no_local_path(blocked_path.output, "blocked shell command output")
        assert_no_local_path(blocked_path.metadata, "blocked shell command metadata")

        with patch("jarvis_v2.tools.shell._run_bounded_process") as run_mock:
            malformed_path = run_shell_command({"command": "python3 '/var/folders/zc/jarvis-shell"})
        if run_mock.called:
            raise SystemExit("Path-shaped malformed shell command must be rejected before spawning.")
        if malformed_path.ok or not malformed_path.metadata.get("parse_error"):
            raise SystemExit(f"Path-shaped malformed shell command should fail safely: {malformed_path.metadata}")
        assert_shell_refusal_handoff(
            malformed_path,
            "parse_error",
            "path-shaped malformed shell command",
        )
        assert_no_local_path(malformed_path.output, "malformed shell command output")
        assert_no_local_path(malformed_path.metadata, "malformed shell command metadata")

        with patch(
            "jarvis_v2.tools.shell._run_bounded_process",
            side_effect=FileNotFoundError("mocked executable not found"),
        ):
            start_failed = run_shell_command({"command": "definitely-not-a-jarvis-command"})
        if start_failed.ok or not start_failed.metadata.get("start_failed"):
            raise SystemExit(f"Missing shell executable should report start failure: {start_failed.metadata}")
        for key in ("execution_started", "action_attempted", "side_effect_possible", "process_spawned"):
            if start_failed.metadata.get(key) is not False:
                raise SystemExit(f"Start-failed shell command should keep {key}=False: {start_failed.metadata}")
        assert_shell_refusal_handoff(start_failed, "start_failed", "start-failed shell command")

        actual_start_failed = run_shell_command({"command": "jarvis-v2-definitely-missing-executable"})
        if actual_start_failed.ok or not actual_start_failed.metadata.get("start_failed"):
            raise SystemExit(f"Actual missing executable did not fail before spawn: {actual_start_failed}")
        for key in ("execution_started", "action_attempted", "side_effect_possible", "process_spawned"):
            if actual_start_failed.metadata.get(key) is not False:
                raise SystemExit(f"Actual start failure should keep {key}=False: {actual_start_failed.metadata}")

        oversized_cwd = "/tmp/" + ("c" * MAX_COMMAND_CHARS)
        with patch("jarvis_v2.tools.shell._run_bounded_process") as cwd_runner:
            cwd_too_long = run_shell_command({"command": "python3 --version", "cwd": oversized_cwd})
        if cwd_runner.called or cwd_too_long.ok or not cwd_too_long.metadata.get("cwd_too_long"):
            raise SystemExit(f"Oversized cwd should fail before spawn: {cwd_too_long}")
        assert_shell_refusal_handoff(cwd_too_long, "cwd_too_long", "oversized shell cwd")

        bad_numbers_completed = mocked_process_outcome(stdout=b"Python mocked\n")
        with patch(
            "jarvis_v2.tools.shell._run_bounded_process",
            return_value=bad_numbers_completed,
        ) as run_mock:
            bad_numbers = run_shell_command(
                {
                    "command": "python3 --version",
                    "timeout": "/var/folders/zc/jarvis-shell-timeout",
                    "max_chars": "/tmp/jarvis-shell-max-chars",
                }
            )
        if run_mock.call_args.args[0] != ["python3", "--version"]:
            raise SystemExit(f"Mocked bad-number command did not receive its exact argv: {run_mock.call_args}")
        if bad_numbers.metadata.get("raw_timeout") != "<local-path>" or bad_numbers.metadata.get("raw_max_chars") != "<local-path>":
            raise SystemExit(f"Shell numeric metadata should redact path-shaped values: {bad_numbers.metadata}")
        assert_no_local_path(bad_numbers.output, "shell command output with path-shaped numeric metadata")
        assert_no_local_path(bad_numbers.metadata, "shell command metadata with path-shaped numeric metadata")


if __name__ == "__main__":
    main()
