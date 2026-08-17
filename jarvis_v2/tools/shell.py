from __future__ import annotations

import os
import selectors
import shlex
import re
import signal
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from jarvis_v2.agent.failure_guidance import declare_failure_guidance
from jarvis_v2.agent.types import ToolResult


MAX_COMMAND_CHARS = 600
MAX_CWD_CHARS = 500
STREAM_READ_CHUNK_BYTES = 64 * 1024
STREAM_BYTES_PER_OUTPUT_CHAR = 4
PROCESS_TERM_GRACE_SECONDS = 0.5
PROCESS_KILL_GRACE_SECONDS = 1.0
POST_STOP_DRAIN_SECONDS = 0.5
LOCAL_PATH_RE = re.compile(r"/(?:Users|private|var/folders|tmp)/[^\n\r;]*")


@dataclass
class _BoundedBytes:
    limit: int
    data: bytearray = field(default_factory=bytearray)
    bytes_seen: int = 0

    def append(self, chunk: bytes) -> None:
        self.bytes_seen += len(chunk)
        remaining = self.limit - len(self.data)
        if remaining > 0:
            self.data.extend(chunk[:remaining])

    @property
    def truncated(self) -> bool:
        return self.bytes_seen > len(self.data)


@dataclass(frozen=True)
class _ProcessOutcome:
    returncode: int | None
    stdout: bytes
    stderr: bytes
    stdout_bytes_seen: int
    stderr_bytes_seen: int
    capture_limit_bytes_per_stream: int
    stdout_truncated: bool = False
    stderr_truncated: bool = False
    streaming_started: bool = True
    timed_out: bool = False
    post_start_failed: bool = False
    exception_type: str | None = None
    termination_escalated: bool = False
    process_reaped: bool = True


def _bounded_int(value: Any, default: int, low: int, high: int) -> int:
    if isinstance(value, bool):
        return default
    try:
        number = int(value)
    except (TypeError, ValueError):
        number = default
    return max(low, min(high, number))


def _raw_int_metadata(value: Any, *, key: str, sanitized: int) -> dict[str, Any]:
    if value is None:
        return {key: sanitized}
    if isinstance(value, bool):
        return {key: sanitized, f"raw_{key}": str(value)}
    try:
        int(value)
    except (TypeError, ValueError):
        return {key: sanitized, f"raw_{key}": _short(value, limit=80)}
    return {key: sanitized}


def _short(value: Any, *, limit: int) -> str:
    text = str(value or "").strip()
    text = " ".join(text.split())
    text = LOCAL_PATH_RE.sub("<local-path>", text)
    if len(text) > limit:
        return text[: limit - 1].rstrip() + "…"
    return text


def _captured_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        text = value.decode("utf-8", errors="replace")
    else:
        text = str(value)
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _bounded_redacted_output(value: str, *, limit: int, force_truncated: bool = False) -> str:
    text = LOCAL_PATH_RE.sub("<local-path>", value)
    if len(text) <= limit and not force_truncated:
        return text
    suffix = "\n... truncated"
    return text[: limit - len(suffix)].rstrip() + suffix


def _command_audit_summary(command: Any) -> str:
    if isinstance(command, (list, tuple)):
        parts = [str(part) for part in command]
    else:
        command_text = str(command or "").strip()
        try:
            parts = shlex.split(command_text)
        except ValueError:
            parts = command_text.split(maxsplit=1)
    if not parts:
        return ""
    executable = _short(parts[0], limit=MAX_COMMAND_CHARS)
    if len(parts) > 1:
        return _bounded_redacted_output(f"{executable} <arguments-redacted>", limit=MAX_COMMAND_CHARS)
    return executable


def _captured_output(*, stdout: Any, stderr: Any, limit: int, truncated: bool = False) -> str:
    output = _captured_text(stdout)
    captured_stderr = _captured_text(stderr)
    if captured_stderr:
        output += ("\n" if output else "") + "stderr:\n" + captured_stderr
    return _bounded_redacted_output(output, limit=limit, force_truncated=truncated).strip()


def _post_start_output(
    message: str,
    *,
    stdout: Any,
    stderr: Any,
    limit: int,
    truncated: bool = False,
) -> str:
    captured = _captured_output(stdout=stdout, stderr=stderr, limit=limit, truncated=truncated)
    output = message + (("\n" + captured) if captured else "")
    return _bounded_redacted_output(output, limit=limit).strip()


def _close_selector_stream(selector: selectors.BaseSelector, fileobj: Any) -> None:
    try:
        selector.unregister(fileobj)
    except Exception:
        pass
    try:
        fileobj.close()
    except Exception:
        pass


def _read_ready_streams(
    selector: selectors.BaseSelector,
    stdout: _BoundedBytes,
    stderr: _BoundedBytes,
    *,
    timeout: float,
) -> Exception | None:
    try:
        events = selector.select(timeout)
    except Exception as exc:
        return exc
    for key, _mask in events:
        try:
            chunk = os.read(key.fd, STREAM_READ_CHUNK_BYTES)
        except BlockingIOError:
            continue
        except Exception as exc:
            _close_selector_stream(selector, key.fileobj)
            return exc
        if not chunk:
            _close_selector_stream(selector, key.fileobj)
            continue
        target = stdout if key.data == "stdout" else stderr
        target.append(chunk)
    return None


def _signal_process_group(process: subprocess.Popen[bytes], sig: int) -> None:
    try:
        os.killpg(process.pid, sig)
        return
    except ProcessLookupError:
        return
    except OSError:
        pass
    try:
        if sig == signal.SIGTERM:
            process.terminate()
        else:
            process.kill()
    except (OSError, ProcessLookupError):
        pass


def _process_group_exists(process: subprocess.Popen[bytes]) -> bool:
    try:
        os.killpg(process.pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _terminate_and_reap(process: subprocess.Popen[bytes]) -> tuple[bool, bool]:
    # start_new_session=True makes the child's pid its process-group id. Signal
    # the group even if the direct child already exited, because descendants may
    # still own the output pipes.
    _signal_process_group(process, signal.SIGTERM)
    escalated = False
    grace_deadline = time.monotonic() + PROCESS_TERM_GRACE_SECONDS
    while time.monotonic() < grace_deadline:
        parent_running = process.poll() is None
        group_running = _process_group_exists(process)
        if not parent_running and not group_running:
            break
        remaining = grace_deadline - time.monotonic()
        if remaining <= 0:
            break
        if parent_running:
            try:
                process.wait(timeout=min(remaining, 0.05))
            except subprocess.TimeoutExpired:
                pass
            except OSError:
                break
        else:
            time.sleep(min(remaining, 0.01))
    if _process_group_exists(process):
        escalated = True
        _signal_process_group(process, signal.SIGKILL)
    if process.poll() is None:
        try:
            process.wait(timeout=PROCESS_KILL_GRACE_SECONDS)
        except (subprocess.TimeoutExpired, OSError):
            pass
    return escalated, process.poll() is not None


def _drain_after_stop(
    selector: selectors.BaseSelector,
    stdout: _BoundedBytes,
    stderr: _BoundedBytes,
) -> None:
    deadline = time.monotonic() + POST_STOP_DRAIN_SECONDS
    while selector.get_map():
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        error = _read_ready_streams(selector, stdout, stderr, timeout=min(remaining, 0.05))
        if error is not None:
            break


def _run_bounded_process(
    command: list[str],
    *,
    cwd: Path,
    timeout: int,
    max_chars: int,
) -> _ProcessOutcome:
    process = subprocess.Popen(
        command,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=False,
        bufsize=0,
        start_new_session=True,
        shell=False,
    )
    capture_limit = max_chars * STREAM_BYTES_PER_OUTPUT_CHAR
    stdout = _BoundedBytes(capture_limit)
    stderr = _BoundedBytes(capture_limit)
    selector: selectors.BaseSelector | None = None
    timed_out = False
    stream_error: Exception | None = None
    streaming_started = False
    termination_escalated = False
    process_reaped = False
    deadline = time.monotonic() + timeout

    try:
        try:
            selector = selectors.DefaultSelector()
            if process.stdout is None or process.stderr is None:
                raise RuntimeError("subprocess output pipes unavailable")
            os.set_blocking(process.stdout.fileno(), False)
            os.set_blocking(process.stderr.fileno(), False)
            selector.register(process.stdout, selectors.EVENT_READ, "stdout")
            selector.register(process.stderr, selectors.EVENT_READ, "stderr")
            streaming_started = True
        except Exception as exc:
            stream_error = exc

        while stream_error is None and selector is not None and selector.get_map():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                timed_out = True
                break
            stream_error = _read_ready_streams(
                selector,
                stdout,
                stderr,
                timeout=min(remaining, 0.25),
            )

        if stream_error is None and not timed_out and process.poll() is None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                timed_out = True
            else:
                try:
                    process.wait(timeout=remaining)
                except subprocess.TimeoutExpired:
                    timed_out = True

        if timed_out or stream_error is not None:
            termination_escalated, process_reaped = _terminate_and_reap(process)
            if selector is not None:
                _drain_after_stop(selector, stdout, stderr)
        else:
            process_reaped = process.poll() is not None
    except Exception as exc:
        stream_error = stream_error or exc
        termination_escalated, process_reaped = _terminate_and_reap(process)
        if selector is not None:
            _drain_after_stop(selector, stdout, stderr)
    finally:
        if selector is not None:
            for key in list(selector.get_map().values()):
                _close_selector_stream(selector, key.fileobj)
            selector.close()
        for stream in (process.stdout, process.stderr):
            try:
                if stream is not None:
                    stream.close()
            except Exception:
                pass

    returncode = process.poll()
    if returncode is None:
        try:
            returncode = process.wait(timeout=0)
        except (subprocess.TimeoutExpired, OSError):
            returncode = None
    process_reaped = process_reaped or process.poll() is not None
    return _ProcessOutcome(
        returncode=returncode,
        stdout=bytes(stdout.data),
        stderr=bytes(stderr.data),
        stdout_bytes_seen=stdout.bytes_seen,
        stderr_bytes_seen=stderr.bytes_seen,
        capture_limit_bytes_per_stream=capture_limit,
        stdout_truncated=stdout.truncated,
        stderr_truncated=stderr.truncated,
        streaming_started=streaming_started,
        timed_out=timed_out,
        post_start_failed=stream_error is not None,
        exception_type=type(stream_error).__name__ if stream_error is not None else None,
        termination_escalated=termination_escalated,
        process_reaped=process_reaped,
    )


def _capture_metadata(outcome: _ProcessOutcome, *, output_truncated: bool) -> dict[str, Any]:
    cleanup_attempted = outcome.timed_out or outcome.post_start_failed
    return {
        "process_spawned": True,
        "output_capture_streaming": outcome.streaming_started,
        "capture_limit_bytes_per_stream": outcome.capture_limit_bytes_per_stream,
        "stdout_bytes_seen": outcome.stdout_bytes_seen,
        "stderr_bytes_seen": outcome.stderr_bytes_seen,
        "stdout_truncated": outcome.stdout_truncated,
        "stderr_truncated": outcome.stderr_truncated,
        "output_truncated": output_truncated,
        "termination_escalated": outcome.termination_escalated,
        "process_reaped": outcome.process_reaped,
        "process_group_cleanup_attempted": cleanup_attempted,
        "descendant_cleanup_scope": "original_process_group" if cleanup_attempted else "not_applicable",
        "all_descendants_terminated_verified": False,
        "detached_descendants_may_survive": True,
    }


def _shell_metadata(**extra: Any) -> dict[str, Any]:
    metadata = {
        "calls_model": False,
        "executes_tools": True,
        "queues_approval": False,
        "approves_request": False,
        "dismisses_request": False,
        "reads_private_data": True,
        "reads_personal_data": True,
        "executes_side_effect": True,
        "writes_files": True,
        "writes_memory": False,
        "writes_notes": False,
        "controls_computer": False,
        "requires_approval": True,
        "approval_gated": True,
    }
    if "command" in extra:
        extra["command"] = _command_audit_summary(extra["command"])
    if "cwd" in extra:
        extra["cwd"] = _bounded_redacted_output(_captured_text(extra["cwd"]), limit=MAX_CWD_CHARS)
    metadata.update(extra)
    return metadata


def _shell_refusal_boundaries() -> dict[str, bool]:
    return {
        "calls_model": False,
        "executes_tools": False,
        "queues_approval": False,
        "approves_request": False,
        "dismisses_request": False,
        "reads_private_data": False,
        "reads_personal_data": False,
        "executes_side_effect": False,
        "writes_files": False,
        "writes_memory": False,
        "writes_notes": False,
        "controls_computer": False,
        "requires_approval": True,
        "approval_gated": True,
    }


def _shell_refusal_handoff(
    *,
    reason: str,
    command: Any,
    cwd: str | None = None,
    exception_type: str | None = None,
    changed: list[str] | None = None,
) -> dict[str, Any]:
    command_preview = _command_audit_summary(command)
    handoff = {
        "source": "run_shell_command",
        "mutation": "shell_command_execute",
        "reason": reason,
        "refused": True,
        "ready_for_operator": True,
        "approval_gated": True,
        "requires_approval": True,
        "command": command_preview,
        "command_preview": command_preview,
        "changed": changed or [],
        "next_commands": [
            "run command <single command with plain arguments>",
            "approval review",
            "safety status",
        ],
        "boundaries": _shell_refusal_boundaries(),
    }
    if cwd is not None:
        handoff["cwd"] = _bounded_redacted_output(cwd, limit=MAX_CWD_CHARS)
    if exception_type:
        handoff["exception_type"] = exception_type
    return handoff


def _shell_refusal_metadata(*, reason: str, command: Any, **extra: Any) -> dict[str, Any]:
    cwd = extra.get("cwd")
    exception_type = extra.get("exception_type")
    handoff = _shell_refusal_handoff(
        reason=reason,
        command=command,
        cwd=cwd if isinstance(cwd, str) else None,
        exception_type=exception_type if isinstance(exception_type, str) else None,
    )
    return _shell_metadata(
        **_shell_refusal_boundaries(),
        outcome_known=True,
        outcome_unknown=False,
        execution_outcome_unknown=False,
        execution_started=False,
        action_attempted=False,
        side_effect_possible=False,
        process_spawned=False,
        retry_safe=False,
        automatic_retry_allowed=False,
        authorizes_retry=False,
        command=handoff["command"],
        shell_refusal_handoff=handoff,
        shell_refusal_handoff_ready=True,
        refusal_reason=reason,
        **extra,
    )


def _shell_failure_guidance(
    metadata: dict[str, Any],
    *,
    output: str,
    action: str,
    outcome_unknown: bool,
    side_effect_possible: bool,
) -> dict[str, Any]:
    guided = dict(metadata)
    guided.update(
        {
            "outcome_known": not outcome_unknown,
            "outcome_unknown": outcome_unknown,
            "execution_outcome_unknown": outcome_unknown,
            "side_effect_possible": side_effect_possible,
            "retry_safe": False,
            "automatic_retry_allowed": False,
            "authorizes_retry": False,
        }
    )
    return declare_failure_guidance(guided, output=output, action=action)


def run_shell_command(args: dict[str, Any]) -> ToolResult:
    raw_command = args.get("command")
    command_text = raw_command if isinstance(raw_command, str) else str(raw_command or "")
    if not command_text.strip():
        output = (
            "No command provided. Provide one concrete command with plain arguments through the normal "
            "HIGH_RISK approval flow."
        )
        action = "Provide one concrete command with plain arguments through the normal HIGH_RISK approval flow."
        return ToolResult(
            "run_shell_command",
            False,
            output,
            _shell_failure_guidance(
                _shell_refusal_metadata(reason="missing_command", command=""),
                output=output,
                action=action,
                outcome_unknown=False,
                side_effect_possible=False,
            ),
        )
    if len(command_text) > MAX_COMMAND_CHARS:
        command_preview = _short(command_text, limit=MAX_COMMAND_CHARS)
        action = f"Command is too long. Maximum length is {MAX_COMMAND_CHARS} characters."
        output = action
        return ToolResult(
            "run_shell_command",
            False,
            output,
            _shell_failure_guidance(
                _shell_refusal_metadata(
                    reason="command_too_long",
                    command=command_preview,
                    command_too_long=True,
                    command_chars=len(command_text),
                    max_command_chars=MAX_COMMAND_CHARS,
                ),
                output=output,
                action=action,
                outcome_unknown=False,
                side_effect_possible=False,
            ),
        )
    blocked_tokens = {";", "&&", "||", "|", ">", ">>", "<", "$(", "`"}
    if any(token in command_text for token in blocked_tokens):
        action = "Submit one command with plain arguments through the normal HIGH_RISK approval flow."
        output = f"Shell control operators are not allowed. {action}"
        return ToolResult(
            "run_shell_command",
            False,
            output,
            _shell_failure_guidance(
                _shell_refusal_metadata(reason="blocked_control_operator", command=command_text, blocked=True),
                output=output,
                action=action,
                outcome_unknown=False,
                side_effect_possible=False,
            ),
        )

    try:
        command = shlex.split(command_text)
    except ValueError as exc:
        action = "Correct the quoting, then submit one command through the normal HIGH_RISK approval flow."
        output = f"Could not parse command. Check quoting. {action}"
        return ToolResult(
            "run_shell_command",
            False,
            output,
            _shell_failure_guidance(
                _shell_refusal_metadata(
                    reason="parse_error",
                    command=command_text,
                    parse_error=True,
                    exception_type=type(exc).__name__,
                ),
                output=output,
                action=action,
                outcome_unknown=False,
                side_effect_possible=False,
            ),
        )
    if not command:
        output = (
            "No command provided. Provide one concrete command with plain arguments through the normal "
            "HIGH_RISK approval flow."
        )
        action = "Provide one concrete command with plain arguments through the normal HIGH_RISK approval flow."
        return ToolResult(
            "run_shell_command",
            False,
            output,
            _shell_failure_guidance(
                _shell_refusal_metadata(reason="missing_command", command=""),
                output=output,
                action=action,
                outcome_unknown=False,
                side_effect_possible=False,
            ),
        )

    cwd_text = str(args.get("cwd") or ".").strip() or "."
    if len(cwd_text) > MAX_CWD_CHARS:
        action = (
            f"Shorten the working directory to at most {MAX_CWD_CHARS} characters, then submit the command through "
            "the normal HIGH_RISK approval flow."
        )
        output = f"Working directory is too long. Maximum length is {MAX_CWD_CHARS} characters. {action}"
        return ToolResult(
            "run_shell_command",
            False,
            output,
            _shell_failure_guidance(
                _shell_refusal_metadata(
                    reason="cwd_too_long",
                    command=command,
                    cwd=_short(cwd_text, limit=MAX_CWD_CHARS),
                    cwd_too_long=True,
                    cwd_chars=len(cwd_text),
                    max_cwd_chars=MAX_CWD_CHARS,
                ),
                output=output,
                action=action,
                outcome_unknown=False,
                side_effect_possible=False,
            ),
        )
    cwd = Path(cwd_text).expanduser()
    timeout = _bounded_int(args.get("timeout"), 30, 1, 120)
    max_chars = _bounded_int(args.get("max_chars"), 6000, 500, 20000)

    try:
        outcome = _run_bounded_process(
            command,
            cwd=cwd,
            timeout=timeout,
            max_chars=max_chars,
        )
    except (FileNotFoundError, PermissionError, NotADirectoryError, IsADirectoryError) as exc:
        action = (
            "Run `setup check`, correct the executable or working-directory issue, then submit a new "
            "HIGH_RISK approval."
        )
        output = f"Command failed to start. {action}"
        return ToolResult(
            "run_shell_command",
            False,
            output,
            _shell_failure_guidance(
                _shell_refusal_metadata(
                    reason="start_failed",
                    command=command,
                    cwd=str(cwd),
                    start_failed=True,
                    exception_type=type(exc).__name__,
                    **_raw_int_metadata(args.get("timeout"), key="timeout", sanitized=timeout),
                    **_raw_int_metadata(args.get("max_chars"), key="max_chars", sanitized=max_chars),
                ),
                output=output,
                action=action,
                outcome_unknown=False,
                side_effect_possible=False,
            ),
        )
    except Exception as exc:
        action = (
            "Command execution outcome is unknown because launch may have begun. Check the intended target state "
            "and audit trail; do not rerun automatically."
        )
        return ToolResult(
            "run_shell_command",
            False,
            action,
            _shell_failure_guidance(
                _shell_metadata(
                    command=command,
                    cwd=str(cwd),
                    post_start_failed=True,
                    execution_started=None,
                    action_attempted=True,
                    process_spawned=None,
                    exception_type=type(exc).__name__,
                    **_raw_int_metadata(args.get("timeout"), key="timeout", sanitized=timeout),
                    **_raw_int_metadata(args.get("max_chars"), key="max_chars", sanitized=max_chars),
                ),
                output=action,
                action=action,
                outcome_unknown=True,
                side_effect_possible=True,
            ),
        )

    combined_unbounded = _captured_text(outcome.stdout)
    captured_stderr = _captured_text(outcome.stderr)
    if captured_stderr:
        combined_unbounded += ("\n" if combined_unbounded else "") + "stderr:\n" + captured_stderr
    redacted_unbounded = LOCAL_PATH_RE.sub("<local-path>", combined_unbounded)
    output_truncated = (
        outcome.stdout_truncated
        or outcome.stderr_truncated
        or len(redacted_unbounded) > max_chars
    )
    capture_metadata = _capture_metadata(outcome, output_truncated=output_truncated)

    if outcome.timed_out:
        action = (
            "Command outcome is unknown after timeout. Inspect the target state and audit trail; do not rerun "
            "automatically."
        )
        output = _post_start_output(
            f"Command timed out after {timeout} seconds. Execution started and may have produced side effects. "
            + action
            + " Jarvis attempted to stop the original process group; independently detached descendants cannot be verified.",
            stdout=outcome.stdout,
            stderr=outcome.stderr,
            limit=max_chars,
            truncated=output_truncated,
        )
        return ToolResult(
            "run_shell_command",
            False,
            output,
            _shell_failure_guidance(
                _shell_metadata(
                    command=command,
                    cwd=str(cwd),
                    timed_out=True,
                    execution_started=True,
                    action_attempted=True,
                    exception_type="TimeoutExpired",
                    returncode=outcome.returncode,
                    **capture_metadata,
                    **_raw_int_metadata(args.get("timeout"), key="timeout", sanitized=timeout),
                    **_raw_int_metadata(args.get("max_chars"), key="max_chars", sanitized=max_chars),
                ),
                output=output,
                action=action,
                outcome_unknown=True,
                side_effect_possible=True,
            ),
        )

    if outcome.post_start_failed:
        action = (
            "Command outcome is unknown after output capture failed. Inspect the target state and audit trail; do "
            "not rerun automatically."
        )
        output = _post_start_output(
            "Command output capture failed after execution started and may have produced side effects. "
            + action
            + " Jarvis attempted to stop the original process group; independently detached descendants cannot be verified.",
            stdout=outcome.stdout,
            stderr=outcome.stderr,
            limit=max_chars,
            truncated=output_truncated,
        )
        return ToolResult(
            "run_shell_command",
            False,
            output,
            _shell_failure_guidance(
                _shell_metadata(
                    command=command,
                    cwd=str(cwd),
                    post_start_failed=True,
                    output_capture_failed=True,
                    execution_started=True,
                    action_attempted=True,
                    exception_type=outcome.exception_type or "OutputCaptureError",
                    returncode=outcome.returncode,
                    **capture_metadata,
                    **_raw_int_metadata(args.get("timeout"), key="timeout", sanitized=timeout),
                    **_raw_int_metadata(args.get("max_chars"), key="max_chars", sanitized=max_chars),
                ),
                output=output,
                action=action,
                outcome_unknown=True,
                side_effect_possible=True,
            ),
        )

    output = _captured_output(
        stdout=outcome.stdout,
        stderr=outcome.stderr,
        limit=max_chars,
        truncated=output_truncated,
    )
    if not output:
        returncode_display = outcome.returncode if outcome.returncode is not None else "unknown"
        output = _bounded_redacted_output(
            f"Command exited with code {returncode_display} and no output.",
            limit=max_chars,
        )

    ok = outcome.returncode == 0
    metadata = _shell_metadata(
        returncode=outcome.returncode,
        command=command,
        cwd=str(cwd),
        execution_started=True,
        action_attempted=True,
        side_effect_possible=True,
        **capture_metadata,
        **_raw_int_metadata(args.get("timeout"), key="timeout", sanitized=timeout),
        **_raw_int_metadata(args.get("max_chars"), key="max_chars", sanitized=max_chars),
    )
    if not ok:
        action = (
            "The command exited unsuccessfully after execution started. Inspect the captured output and target "
            "state; do not rerun automatically."
        )
        output = _post_start_output(
            action,
            stdout=outcome.stdout,
            stderr=outcome.stderr,
            limit=max_chars,
            truncated=output_truncated,
        )
        metadata = _shell_failure_guidance(
            metadata,
            output=output,
            action=action,
            outcome_unknown=False,
            side_effect_possible=True,
        )
    return ToolResult(
        "run_shell_command",
        ok,
        output.strip(),
        metadata,
    )
