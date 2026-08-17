from __future__ import annotations

import uuid
from pathlib import Path
from tempfile import TemporaryDirectory

from jarvis_v2.agent.planner import RuleBasedPlanner
from jarvis_v2.scripts.test_runtime import handle_runtime_case, make_temp_runtime
from jarvis_v2.tools import files
from jarvis_v2.tools import shell
from jarvis_v2.tools import system
from jarvis_v2.tools import utilities


def assert_operator_limits(metadata: dict, output: str, label: str) -> None:
    if metadata.get("operator_timeboxes_override_priority") is not True:
        raise SystemExit(f"{label} missed operator timebox metadata: {metadata}")
    if metadata.get("stop_times_override_priority") is not True:
        raise SystemExit(f"{label} missed stop-time metadata: {metadata}")
    if "explicit stop times" not in output or "pause commands" not in output:
        raise SystemExit(f"{label} missed operator-limit output.")


def assert_utility_handoff(result, *, label: str, source: str, status: str, reason: str | None = None) -> None:
    metadata = result.metadata
    handoff = metadata.get("utility_handoff")
    if not metadata.get("utility_handoff_ready") or not isinstance(handoff, dict):
        raise SystemExit(f"{label} should expose utility_handoff: {metadata}")
    if handoff.get("source") != source or handoff.get("status") != status:
        raise SystemExit(f"{label} utility handoff source/status mismatch: {handoff}")
    if metadata.get("utility_status") != status:
        raise SystemExit(f"{label} utility status should mirror handoff: {metadata}")
    if handoff.get("changed") != [] or not handoff.get("ready_for_operator"):
        raise SystemExit(f"{label} should report no-change/operator-ready state: {handoff}")
    if reason:
        if handoff.get("reason") != reason or metadata.get("refusal_reason") != reason or handoff.get("refused") is not True:
            raise SystemExit(f"{label} utility refusal reason mismatch: {metadata}")
    elif handoff.get("refused") is not False:
        raise SystemExit(f"{label} utility success should not be marked refused: {handoff}")
    boundaries = handoff.get("boundaries") or {}
    for key in (
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
        "external_side_effect",
        "authorizes_execution",
        "authorizes_completion_claim",
        "approval_granted",
    ):
        if metadata.get(key) or boundaries.get(key):
            raise SystemExit(f"{label} should keep {key}=False: {metadata}")
    for key in ("authorizes_execution", "authorizes_completion_claim", "approval_granted"):
        if handoff.get(key) is not False:
            raise SystemExit(f"{label} utility handoff should keep {key}=False: {handoff}")
    if boundaries.get("read_only") is not True:
        raise SystemExit(f"{label} utility handoff should be read-only: {handoff}")
    if source in {"generate_password", "generate_uuid"} and str(result.output) in str(handoff):
        raise SystemExit(f"{label} utility handoff should not include generated secret/id content: {handoff}")
    if any(fragment in str(handoff) for fragment in ("/\x55sers/", "/private/", "/var/folders/", "/tmp/")):
        raise SystemExit(f"{label} utility handoff leaked a local path: {handoff}")


def assert_system_handoff(
    result,
    *,
    label: str,
    source: str,
    status: str,
    reason: str | None = None,
    state_changed: bool = False,
    changed: list[str] | None = None,
) -> None:
    metadata = result.metadata
    handoff = metadata.get("system_handoff")
    expected_changed = list(changed or []) if state_changed else []
    if metadata.get("system_handoff_ready") is not True or not isinstance(handoff, dict):
        raise SystemExit(f"{label} should expose system_handoff: {metadata}")
    if handoff.get("source") != source or handoff.get("status") != status or metadata.get("system_status") != status:
        raise SystemExit(f"{label} system handoff source/status mismatch: {handoff} / {metadata}")
    if reason:
        if handoff.get("reason") != reason or metadata.get("refusal_reason") != reason or handoff.get("refused") is not True:
            raise SystemExit(f"{label} system refusal reason mismatch: {handoff} / {metadata}")
    elif handoff.get("refused") is not False:
        raise SystemExit(f"{label} system handoff should not be refused: {handoff}")
    if handoff.get("ready_for_operator") is not True or handoff.get("content_in_handoff") is not False:
        raise SystemExit(f"{label} system handoff should be content-free and operator-ready: {handoff}")
    for key in ("state_changed", "system_state_changed"):
        actual = handoff.get(key) if key == "state_changed" else metadata.get(key)
        if actual is not state_changed:
            raise SystemExit(f"{label} system change truth mismatch for {key}: {handoff} / {metadata}")
    for key in ("changed", "system_changed"):
        actual = handoff.get(key) if key == "changed" else metadata.get(key)
        if actual != expected_changed:
            raise SystemExit(f"{label} system change identities mismatch for {key}: {handoff} / {metadata}")
    if metadata.get("state_changed") != handoff.get("state_changed") or metadata.get("changed") != handoff.get("changed"):
        raise SystemExit(f"{label} system top-level/handoff change parity failed: {handoff} / {metadata}")
    commands = handoff.get("next_commands")
    if not isinstance(commands, list) or "system info" not in commands or "what volume" not in commands:
        raise SystemExit(f"{label} system handoff missed recovery commands: {handoff}")
    boundaries = handoff.get("boundaries")
    if not isinstance(boundaries, dict):
        raise SystemExit(f"{label} system handoff missed boundaries: {handoff}")
    for key in (
        "calls_model",
        "calls_external_service",
        "executes_tools",
        "queues_approval",
        "requires_approval",
        "approves_request",
        "dismisses_request",
        "external_side_effect",
        "writes_files",
        "writes_database",
        "writes_memory",
        "writes_notes",
        "controls_computer",
        "speaks",
        "completes_tasks",
        "authorizes_execution",
        "authorizes_completion_claim",
        "approval_granted",
    ):
        if boundaries.get(key):
            raise SystemExit(f"{label} system boundary should keep {key}=False: {handoff}")
        if metadata.get(key):
            raise SystemExit(f"{label} system metadata should keep {key}=False: {metadata}")
    for key in ("authorizes_execution", "authorizes_completion_claim", "approval_granted"):
        if handoff.get(key) is not False:
            raise SystemExit(f"{label} system handoff should keep {key}=False: {handoff}")
    for key in ("app", "level", "available", "chars", "max_chars", "truncated", "returncode", "error_type"):
        if key in handoff and handoff.get(key) != metadata.get(key):
            raise SystemExit(f"{label} system handoff {key} parity failed: {handoff} / {metadata}")
    if source == "get_clipboard" and "private clipboard text" in str(handoff):
        raise SystemExit(f"{label} system handoff leaked clipboard content: {handoff}")
    if source == "set_clipboard" and "xxxxxxxxxx" in str(handoff):
        raise SystemExit(f"{label} system handoff leaked copied clipboard content: {handoff}")
    if any(fragment in str(handoff) for fragment in ("/\x55sers/", "/private/", "/var/folders/", "/tmp/")):
        raise SystemExit(f"{label} system handoff leaked a local path: {handoff}")


def main() -> None:
    with TemporaryDirectory(prefix="jarvis-functions-") as temp:
        root = Path(temp)
        runtime = make_temp_runtime(root)
        test_path = root / "tmp_v2_function_test.md"
        large_path = root / "large_private_note.md"
        large_path.write_text("x" * (files.MAX_READ_BYTES + 1), encoding="utf-8")

        cases = [
            ("calculate 12 * (4 + 3)", False, True),
            ("divide 84 by 3", False, True),
            ("12 times 8", False, True),
            ("subtract 9 from 20", False, True),
            ("what is 18% of 240", False, True),
            ("what is 20 percent tip on 50", False, True),
            ("split 50 dollars between 2 people", False, True),
            ("spell restaurant", False, True),
            ("word count hello world", False, True),
            ("uppercase hello world", False, True),
            ("slugify hello world", False, True),
            ("repeat hello 3 times", False, True),
            ("convert 10 km to miles", False, True),
            ("what is 10 pounds in kg", False, True),
            ("how many cups in a liter", False, True),
            ("generate password 12", False, True),
            ("generate a 16 character password", False, True),
            ("generate a 16 character password without symbols", False, True),
            ("generate uuid", False, True),
            ("password 12 characters", False, True),
            ("list files in .", False, False),
            ("list files in .", True, True),
            ("find files README in .", False, False),
            ("find files README in .", True, True),
            ("system info", False, True),
            ("what volume", False, True),
            ("list tools", False, True),
            (f"write file {test_path} with Jarvis V2 function smoke test.", False, False),
            (f"write file {test_path} with Jarvis V2 function smoke test.", True, True),
            (f"read file {test_path}", False, False),
            (f"read file {test_path}", True, True),
        ]

        for case, approved, expected_verified in cases:
            result = handle_runtime_case(runtime, case, approved=approved)
            status = "ok" if result.verified else "blocked"
            approval = " approved" if approved else ""
            print(f"[{status}{approval}] {case}")
            print(result.response)
            print()
            if expected_verified is None:
                continue
            if result.verified is not expected_verified:
                raise SystemExit(f"Unexpected verification state for {case!r}: {result.verified}")
            if not approved and not expected_verified and "explicit approval required" not in result.response:
                raise SystemExit(f"Blocked command did not explain approval requirement: {case}")
            if case == "what volume":
                metadata = result.tool_results[0].metadata
                if "Current volume" not in result.response:
                    raise SystemExit("Volume read should return a stable current-volume response or unavailable notice.")
                if metadata.get("writes_files") is not False or metadata.get("controls_computer") is not False or metadata.get("queues_approval"):
                    raise SystemExit("Volume read missed safety metadata.")
                assert_operator_limits(metadata, result.response, "Volume read")
            if case == "system info":
                metadata = result.tool_results[0].metadata
                if metadata.get("reads_system_info") is not True:
                    raise SystemExit("system_info missed system-info metadata.")
                assert_operator_limits(metadata, result.response, "system_info routed")

        direct_list = files.list_files({"directory": root, "limit": "9999"})
        if not direct_list.ok or direct_list.metadata.get("limit") != files.MAX_LIST_LIMIT:
            raise SystemExit("list_files did not clamp large limits.")
        if direct_list.metadata.get("reads_private_data") is not True or direct_list.metadata.get("writes_files") is not False:
            raise SystemExit("list_files missed private-data metadata.")
        if direct_list.metadata.get("queues_approval") is not False or direct_list.metadata.get("controls_computer") is not False:
            raise SystemExit("list_files missed safety metadata.")

        bad_list = files.list_files({"directory": root, "limit": "bad"})
        if not bad_list.ok or bad_list.metadata.get("limit") != 80 or bad_list.metadata.get("raw_limit") != "bad":
            raise SystemExit("list_files did not sanitize bad limits.")
        bool_list = files.list_files({"directory": root, "limit": False})
        if not bool_list.ok or bool_list.metadata.get("limit") != 80 or bool_list.metadata.get("raw_limit") != "False":
            raise SystemExit(f"list_files should treat boolean limits as malformed and preserve raw metadata: {bool_list.metadata}")
        bad_list_long = files.list_files({"directory": root, "limit": "l" * 200})
        if bad_list_long.metadata.get("raw_limit") != ("l" * 79 + "…"):
            raise SystemExit(f"list_files did not bound raw bad limit metadata: {bad_list_long.metadata}")
        path_bad_list = files.list_files({"directory": root, "limit": "/\x55sers/example/private/list-limit"})
        if path_bad_list.metadata.get("raw_limit") != "<local-path>":
            raise SystemExit(f"list_files leaked local path in raw limit metadata: {path_bad_list.metadata}")
        missing_list = files.list_files({"directory": root / "missing", "limit": "9999"})
        if missing_list.ok or missing_list.metadata.get("limit") != files.MAX_LIST_LIMIT:
            raise SystemExit("list_files missing-dir refusal missed bounded metadata.")

        direct_read = files.read_text_file({"path": test_path, "max_chars": "999999"})
        if not direct_read.ok or direct_read.metadata.get("max_chars") != files.MAX_READ_CHARS:
            raise SystemExit("read_text_file did not clamp large max_chars.")
        if direct_read.metadata.get("reads_private_data") is not True or direct_read.metadata.get("writes_files") is not False:
            raise SystemExit("read_text_file missed private-data metadata.")
        if direct_read.metadata.get("queues_approval") is not False or direct_read.metadata.get("controls_computer") is not False:
            raise SystemExit("read_text_file missed safety metadata.")

        bad_read = files.read_text_file({"path": test_path, "max_chars": "bad"})
        if not bad_read.ok or bad_read.metadata.get("max_chars") != 5000 or bad_read.metadata.get("raw_max_chars") != "bad":
            raise SystemExit("read_text_file did not sanitize bad max_chars.")
        bool_read = files.read_text_file({"path": test_path, "max_chars": True})
        if not bool_read.ok or bool_read.metadata.get("max_chars") != 5000 or bool_read.metadata.get("raw_max_chars") != "True":
            raise SystemExit(f"read_text_file should treat boolean max_chars as malformed and preserve raw metadata: {bool_read.metadata}")
        bad_read_long = files.read_text_file({"path": test_path, "max_chars": "m" * 200})
        if bad_read_long.metadata.get("raw_max_chars") != ("m" * 79 + "…"):
            raise SystemExit(f"read_text_file did not bound raw bad max_chars metadata: {bad_read_long.metadata}")
        path_bad_read = files.read_text_file({"path": test_path, "max_chars": "/private/tmp/jarvis-read-max-chars"})
        if path_bad_read.metadata.get("raw_max_chars") != "<local-path>":
            raise SystemExit(f"read_text_file leaked local path in raw max_chars metadata: {path_bad_read.metadata}")
        missing_read = files.read_text_file({"path": root / "missing.md", "max_chars": "999999"})
        if missing_read.ok or missing_read.metadata.get("max_chars") != files.MAX_READ_CHARS:
            raise SystemExit("read_text_file missing-file refusal missed bounded metadata.")

        large_read = files.read_text_file({"path": large_path})
        if large_read.ok or "Refusing to read large file" not in large_read.output:
            raise SystemExit("read_text_file did not refuse oversized text files.")
        if large_read.metadata.get("max_read_bytes") != files.MAX_READ_BYTES:
            raise SystemExit("large read refusal missed max_read_bytes metadata.")

        class FakeReadPath:
            suffix = ".md"
            name = "secret.md"

            def __init__(self, *, fail: str):
                self.fail = fail

            def __str__(self):
                return "/\x55sers/example/private/secret.md"

            def __fspath__(self):
                return str(self)

            def exists(self):
                return True

            def is_file(self):
                return True

            def stat(self):
                if self.fail == "stat":
                    raise OSError("stat denied for /\x55sers/example/private/secret.md")

                class Stat:
                    st_size = 12

                return Stat()

            def read_text(self, **_kwargs):
                raise OSError("read denied for /\x55sers/example/private/secret.md")

        original_expand_path = files.expand_path
        try:
            files.expand_path = lambda _value: FakeReadPath(fail="stat")  # type: ignore[assignment]
            stat_failed_read = files.read_text_file({"path": "secret.md", "max_chars": "bad"})
            files.expand_path = lambda _value: FakeReadPath(fail="read")  # type: ignore[assignment]
            content_failed_read = files.read_text_file({"path": "secret.md", "max_chars": "bad"})
        finally:
            files.expand_path = original_expand_path  # type: ignore[assignment]
        for label, failed_read in {"stat": stat_failed_read, "read": content_failed_read}.items():
            if failed_read.ok or "Could not" not in failed_read.output or "file." not in failed_read.output:
                raise SystemExit(f"read_text_file {label} failure should return friendly output: {failed_read.output}")
            for expected in [
                "Finder > Get Info > Sharing & Permissions",
                "read permission",
                "setup check",
                "then retry",
            ]:
                if expected not in failed_read.output:
                    raise SystemExit(f"read_text_file {label} failure missed actionable guidance {expected}: {failed_read.output}")
            if "/\x55sers/operator" in failed_read.output or "denied" in failed_read.output:
                raise SystemExit(f"read_text_file {label} failure leaked raw exception text: {failed_read.output}")
            if failed_read.metadata.get("exception_type") != "OSError" or failed_read.metadata.get("raw_max_chars") != "bad":
                raise SystemExit(f"read_text_file {label} failure missed diagnostic metadata: {failed_read.metadata}")

        direct_find = files.find_files({"root": root, "pattern": ".md", "limit": "9999"})
        if not direct_find.ok or direct_find.metadata.get("limit") != files.MAX_FIND_LIMIT:
            raise SystemExit("find_files did not clamp large limits.")
        if direct_find.metadata.get("queues_approval") is not False or direct_find.metadata.get("controls_computer") is not False:
            raise SystemExit("find_files missed safety metadata.")

        bad_find = files.find_files({"root": root, "pattern": ".md", "limit": "bad"})
        if not bad_find.ok or bad_find.metadata.get("limit") != 50 or bad_find.metadata.get("raw_limit") != "bad":
            raise SystemExit("find_files did not sanitize bad limits.")
        bool_find = files.find_files({"root": root, "pattern": ".md", "limit": False})
        if not bool_find.ok or bool_find.metadata.get("limit") != 50 or bool_find.metadata.get("raw_limit") != "False":
            raise SystemExit(f"find_files should treat boolean limits as malformed and preserve raw metadata: {bool_find.metadata}")
        path_bad_find = files.find_files({"root": root, "pattern": ".md", "limit": "/\x55sers/example/private/find-limit"})
        if path_bad_find.metadata.get("raw_limit") != "<local-path>":
            raise SystemExit(f"find_files leaked local path in raw limit metadata: {path_bad_find.metadata}")
        empty_find = files.find_files({"root": root, "pattern": "", "limit": "bad"})
        if empty_find.ok or empty_find.metadata.get("raw_pattern") != "" or empty_find.metadata.get("raw_limit") != "bad":
            raise SystemExit(f"find_files empty pattern should preserve bounded raw pattern/limit metadata: {empty_find.metadata}")
        long_find = files.find_files({"root": root, "pattern": "x" * 1000, "limit": 2})
        if len(long_find.metadata.get("pattern", "")) > files.MAX_PATTERN_CHARS:
            raise SystemExit("find_files did not bound long patterns.")

        direct_write = files.write_text_file({"path": root / "direct_write.md", "content": "safe metadata check"})
        if not direct_write.ok or direct_write.metadata.get("writes_files") is not True:
            raise SystemExit("write_text_file missed writes_files metadata.")
        if direct_write.metadata.get("reads_private_data") is not False or direct_write.metadata.get("controls_computer") is not False:
            raise SystemExit("write_text_file missed safety metadata.")
        if direct_write.metadata.get("queues_approval") is not False:
            raise SystemExit("write_text_file should not mark queued approval when called directly.")

        oversized_write = files.write_text_file({"path": root / "too_large.md", "content": "x" * (files.MAX_WRITE_CHARS + 1)})
        if oversized_write.ok or "oversized" not in oversized_write.output:
            raise SystemExit("write_text_file did not refuse oversized content.")
        if oversized_write.metadata.get("max_chars") != files.MAX_WRITE_CHARS or oversized_write.metadata.get("writes_files"):
            raise SystemExit("write_text_file oversized refusal missed safety metadata.")
        existing_write = files.write_text_file({"path": root / "direct_write.md", "content": "new text"})
        if existing_write.ok or existing_write.metadata.get("exists") is not True or existing_write.metadata.get("writes_files"):
            raise SystemExit("write_text_file existing-file refusal missed metadata.")

        def fail_atomic_write(_path, _content, *, overwrite):
            raise files._TextFilePublicationError(
                OSError("write denied for /\x55sers/example/private/secret.md"),
                committed=False,
            )

        original_atomic_write = files._durable_atomic_write_text
        try:
            files._durable_atomic_write_text = fail_atomic_write  # type: ignore[assignment]
            failed_write = files.write_text_file({"path": root / "secret.md", "content": "nope"})
        finally:
            files._durable_atomic_write_text = original_atomic_write  # type: ignore[assignment]
        if failed_write.ok or "Could not write file." not in failed_write.output:
            raise SystemExit(f"write_text_file failure should return friendly output: {failed_write.output}")
        for expected in [
            "Finder > Get Info > Sharing & Permissions",
            "write permission",
            "setup check",
            "then retry",
        ]:
            if expected not in failed_write.output:
                raise SystemExit(f"write_text_file failure missed actionable guidance {expected}: {failed_write.output}")
        if "/\x55sers/operator" in failed_write.output or "denied" in failed_write.output:
            raise SystemExit(f"write_text_file failure leaked raw exception text: {failed_write.output}")
        if failed_write.metadata.get("exception_type") != "OSError" or failed_write.metadata.get("writes_files"):
            raise SystemExit(f"write_text_file failure missed diagnostic metadata: {failed_write.metadata}")

        direct_shell = shell.run_shell_command({"command": "python3 --version", "timeout": "bad", "max_chars": "10"})
        if not direct_shell.ok or "Python" not in direct_shell.output:
            raise SystemExit("run_shell_command direct smoke failed.")
        if direct_shell.metadata.get("timeout") != 30 or direct_shell.metadata.get("max_chars") != 500:
            raise SystemExit("run_shell_command did not sanitize timeout/max_chars.")
        if direct_shell.metadata.get("executes_tools") is not True:
            raise SystemExit("run_shell_command missed execution metadata.")

        original_system_run = system._run
        try:
            def raise_apps_failure(_args):
                raise RuntimeError("osascript failed near /\x55sers/example/private/window")

            system._run = raise_apps_failure  # type: ignore[assignment]
            app_exception = system.list_running_apps({})
        finally:
            system._run = original_system_run  # type: ignore[assignment]
        if app_exception.ok or "Could not list apps." not in app_exception.output:
            raise SystemExit(f"list_running_apps exception should return friendly output: {app_exception.output}")
        if "/\x55sers/operator" in app_exception.output or "osascript failed" in app_exception.output:
            raise SystemExit(f"list_running_apps exception leaked raw text: {app_exception.output}")
        if app_exception.metadata.get("exception_type") != "RuntimeError" or app_exception.metadata.get("reads_private_data") is not True:
            raise SystemExit(f"list_running_apps exception missed diagnostic/privacy metadata: {app_exception.metadata}")
        assert_operator_limits(app_exception.metadata, app_exception.output, "list_running_apps exception")

        class FakeAppsFailure:
            returncode = 1
            stdout = ""
            stderr = "System Events denied /\x55sers/example/private/window"

        try:
            system._run = lambda _args: FakeAppsFailure()  # type: ignore[assignment]
            app_nonzero = system.list_running_apps({})
        finally:
            system._run = original_system_run  # type: ignore[assignment]
        if app_nonzero.ok or "Could not list apps." not in app_nonzero.output:
            raise SystemExit(f"list_running_apps nonzero should return friendly output: {app_nonzero.output}")
        if "/\x55sers/operator" in app_nonzero.output or "System Events denied" in app_nonzero.output:
            raise SystemExit(f"list_running_apps nonzero leaked raw stderr: {app_nonzero.output}")
        if app_nonzero.metadata.get("returncode") != 1 or app_nonzero.metadata.get("reads_private_data") is not True:
            raise SystemExit(f"list_running_apps nonzero missed diagnostic/privacy metadata: {app_nonzero.metadata}")
        assert_operator_limits(app_nonzero.metadata, app_nonzero.output, "list_running_apps nonzero")

        bad_shell_limits = shell.run_shell_command({"command": "python3 --version", "timeout": "bad", "max_chars": "bad"})
        if not bad_shell_limits.ok or "Python" not in bad_shell_limits.output:
            raise SystemExit("run_shell_command bad-limit direct smoke failed.")
        if bad_shell_limits.metadata.get("timeout") != 30 or bad_shell_limits.metadata.get("raw_timeout") != "bad":
            raise SystemExit(f"run_shell_command should preserve sanitized raw timeout metadata: {bad_shell_limits.metadata}")
        if bad_shell_limits.metadata.get("max_chars") != 6000 or bad_shell_limits.metadata.get("raw_max_chars") != "bad":
            raise SystemExit(f"run_shell_command should preserve sanitized raw max_chars metadata: {bad_shell_limits.metadata}")
        long_bad_shell_limits = shell.run_shell_command({"command": "python3 --version", "timeout": "t" * 200, "max_chars": "m" * 200})
        if long_bad_shell_limits.metadata.get("raw_timeout") != ("t" * 79 + "…"):
            raise SystemExit(f"run_shell_command should bound long raw timeout metadata: {long_bad_shell_limits.metadata}")
        if long_bad_shell_limits.metadata.get("raw_max_chars") != ("m" * 79 + "…"):
            raise SystemExit(f"run_shell_command should bound long raw max_chars metadata: {long_bad_shell_limits.metadata}")
        path_bad_shell_limits = shell.run_shell_command({
            "command": "python3 --version",
            "timeout": "/\x55sers/example/private/shell-timeout",
            "max_chars": "/private/tmp/jarvis-shell-max-chars",
        })
        if "/\x55sers/" in str(path_bad_shell_limits.metadata) or "/private/" in str(path_bad_shell_limits.metadata):
            raise SystemExit(f"run_shell_command should redact local paths in malformed limit metadata: {path_bad_shell_limits.metadata}")
        if path_bad_shell_limits.metadata.get("raw_timeout") != "<local-path>" or path_bad_shell_limits.metadata.get("raw_max_chars") != "<local-path>":
            raise SystemExit(f"run_shell_command should preserve redaction markers for malformed path-like limits: {path_bad_shell_limits.metadata}")
        bool_shell_limits = shell.run_shell_command({"command": "python3 --version", "timeout": True, "max_chars": False})
        if not bool_shell_limits.ok or "Python" not in bool_shell_limits.output:
            raise SystemExit("run_shell_command boolean-limit direct smoke failed.")
        if bool_shell_limits.metadata.get("timeout") != 30 or bool_shell_limits.metadata.get("raw_timeout") != "True":
            raise SystemExit(f"run_shell_command should treat boolean timeout as malformed default: {bool_shell_limits.metadata}")
        if bool_shell_limits.metadata.get("max_chars") != 6000 or bool_shell_limits.metadata.get("raw_max_chars") != "False":
            raise SystemExit(f"run_shell_command should treat boolean max_chars as malformed default: {bool_shell_limits.metadata}")

        blocked_shell = shell.run_shell_command({"command": "python3 --version && whoami"})
        if blocked_shell.ok or "Shell control operators are not allowed" not in blocked_shell.output:
            raise SystemExit("run_shell_command did not block shell control operators.")

        original_shell_run = shell._run_bounded_process
        try:
            def raise_start_failure(*_args, **_kwargs):
                raise FileNotFoundError("missing binary at /\x55sers/example/private/tool")

            shell._run_bounded_process = raise_start_failure  # type: ignore[assignment]
            start_failed_shell = shell.run_shell_command({"command": "missing-tool --version", "timeout": "bad"})
        finally:
            shell._run_bounded_process = original_shell_run  # type: ignore[assignment]
        if start_failed_shell.ok or "Command failed to start." not in start_failed_shell.output:
            raise SystemExit("run_shell_command start failure did not return friendly output.")
        if "/\x55sers/operator" in start_failed_shell.output or "missing binary" in start_failed_shell.output:
            raise SystemExit(f"run_shell_command leaked raw start failure text: {start_failed_shell.output}")
        if start_failed_shell.metadata.get("exception_type") != "FileNotFoundError" or start_failed_shell.metadata.get("start_failed") is not True:
            raise SystemExit(f"run_shell_command start failure missed diagnostic metadata: {start_failed_shell.metadata}")

        class FakeCompleted:
            returncode = 0
            stdout = "private clipboard text"
            stderr = ""

        original_run = system._run
        try:
            system._run = lambda _args: FakeCompleted()  # type: ignore[assignment]
            direct_clipboard = system.get_clipboard({"max_chars": "bad"})
        finally:
            system._run = original_run  # type: ignore[assignment]
        if not direct_clipboard.ok or direct_clipboard.metadata.get("max_chars") != 1000:
            raise SystemExit("get_clipboard did not sanitize max_chars.")
        if direct_clipboard.metadata.get("raw_max_chars") != "bad":
            raise SystemExit(f"get_clipboard should preserve sanitized raw max_chars metadata: {direct_clipboard.metadata}")
        if direct_clipboard.metadata.get("reads_private_data") is not True or direct_clipboard.metadata.get("reads_clipboard") is not True or direct_clipboard.metadata.get("writes_files") is not False:
            raise SystemExit("get_clipboard missed privacy metadata.")
        if direct_clipboard.metadata.get("queues_approval") or direct_clipboard.metadata.get("controls_computer"):
            raise SystemExit("get_clipboard missed approval/computer safety metadata.")
        assert_operator_limits(direct_clipboard.metadata, direct_clipboard.output, "get_clipboard")
        assert_system_handoff(direct_clipboard, label="get_clipboard", source="get_clipboard", status="ok")

        try:
            system._run = lambda _args: FakeCompleted()  # type: ignore[assignment]
            long_bad_clipboard = system.get_clipboard({"max_chars": "m" * 200})
        finally:
            system._run = original_run  # type: ignore[assignment]
        if long_bad_clipboard.metadata.get("raw_max_chars") != ("m" * 79 + "…"):
            raise SystemExit(f"get_clipboard should bound long raw max_chars metadata: {long_bad_clipboard.metadata}")
        assert_operator_limits(long_bad_clipboard.metadata, long_bad_clipboard.output, "get_clipboard long bad max_chars")
        assert_system_handoff(long_bad_clipboard, label="get_clipboard long bad max_chars", source="get_clipboard", status="ok")

        try:
            system._run = lambda _args: FakeCompleted()  # type: ignore[assignment]
            bool_clipboard = system.get_clipboard({"max_chars": True})
        finally:
            system._run = original_run  # type: ignore[assignment]
        if not bool_clipboard.ok or bool_clipboard.metadata.get("max_chars") != 1000:
            raise SystemExit(f"get_clipboard should treat boolean max_chars as malformed defaults: {bool_clipboard.metadata}")
        if bool_clipboard.metadata.get("raw_max_chars") != "True":
            raise SystemExit(f"get_clipboard should preserve boolean raw max_chars metadata: {bool_clipboard.metadata}")
        assert_operator_limits(bool_clipboard.metadata, bool_clipboard.output, "get_clipboard boolean max_chars")
        assert_system_handoff(bool_clipboard, label="get_clipboard boolean max_chars", source="get_clipboard", status="ok")

        try:
            system._run = lambda _args: FakeCompleted()  # type: ignore[assignment]
            path_bad_clipboard = system.get_clipboard({"max_chars": "/private/tmp/jarvis-clipboard-max"})
        finally:
            system._run = original_run  # type: ignore[assignment]
        if not path_bad_clipboard.ok or path_bad_clipboard.metadata.get("raw_max_chars") != "<local-path>":
            raise SystemExit(f"get_clipboard should redact path-shaped raw max_chars metadata: {path_bad_clipboard.metadata}")
        assert_operator_limits(path_bad_clipboard.metadata, path_bad_clipboard.output, "get_clipboard path bad max_chars")
        assert_system_handoff(path_bad_clipboard, label="get_clipboard path bad max_chars", source="get_clipboard", status="ok")

        def assert_recovery_guidance(output: str, label: str, fragments: tuple[str, ...]) -> None:
            for fragment in fragments:
                if fragment not in output:
                    raise SystemExit(f"{label} missed actionable recovery guidance {fragment!r}: {output}")

        class FakeClipboardFailure:
            returncode = 1
            stdout = ""
            stderr = "pbpaste denied /\x55sers/example/private/clipboard"

        try:
            system._run = lambda _args: FakeClipboardFailure()  # type: ignore[assignment]
            failed_clipboard = system.get_clipboard({"max_chars": "bad"})
        finally:
            system._run = original_run  # type: ignore[assignment]
        if failed_clipboard.ok or "Could not read clipboard." not in failed_clipboard.output:
            raise SystemExit(f"get_clipboard failure should return friendly output: {failed_clipboard.output}")
        if "/\x55sers/operator" in failed_clipboard.output or "pbpaste denied" in failed_clipboard.output:
            raise SystemExit(f"get_clipboard failure leaked raw stderr: {failed_clipboard.output}")
        if failed_clipboard.metadata.get("returncode") != 1 or failed_clipboard.metadata.get("raw_max_chars") != "bad":
            raise SystemExit(f"get_clipboard failure missed diagnostic metadata: {failed_clipboard.metadata}")
        assert_recovery_guidance(
            failed_clipboard.output,
            "get_clipboard failure",
            ("System Settings", "Privacy & Security", "Automation", "setup check", "retry"),
        )
        assert_operator_limits(failed_clipboard.metadata, failed_clipboard.output, "get_clipboard failure")
        assert_system_handoff(failed_clipboard, label="get_clipboard failure", source="get_clipboard", status="error")

        large_clipboard = system.set_clipboard({"text": "x" * (system.MAX_CLIPBOARD_WRITE_CHARS + 1)})
        if large_clipboard.ok or "Refusing to copy" not in large_clipboard.output:
            raise SystemExit("set_clipboard did not refuse oversized clipboard writes.")
        if large_clipboard.metadata.get("max_chars") != system.MAX_CLIPBOARD_WRITE_CHARS:
            raise SystemExit("set_clipboard oversized refusal missed max_chars metadata.")
        if large_clipboard.metadata.get("queues_approval") or large_clipboard.metadata.get("writes_files"):
            raise SystemExit("set_clipboard oversized refusal missed safety metadata.")
        assert_operator_limits(large_clipboard.metadata, large_clipboard.output, "set_clipboard oversized refusal")
        assert_system_handoff(large_clipboard, label="set_clipboard oversized refusal", source="set_clipboard", status="refused", reason="too_large")

        clipboard_calls: list[tuple[list[str], object]] = []

        class FakeClipboardProcess:
            def __init__(self, command: list[str], *, stdin, returncode: int) -> None:
                clipboard_calls.append((command, stdin))
                self.returncode = returncode
                self.input_bytes = b""

            def communicate(self, input_bytes: bytes) -> None:
                self.input_bytes = input_bytes

        original_popen = system.subprocess.Popen
        successful_clipboard_process: FakeClipboardProcess | None = None
        try:
            def successful_popen(command, *, stdin):
                nonlocal successful_clipboard_process
                successful_clipboard_process = FakeClipboardProcess(command, stdin=stdin, returncode=0)
                return successful_clipboard_process

            system.subprocess.Popen = successful_popen  # type: ignore[assignment]
            successful_clipboard = system.set_clipboard({"text": "mock clipboard text"})
        finally:
            system.subprocess.Popen = original_popen  # type: ignore[assignment]
        if clipboard_calls != [(["pbcopy"], system.subprocess.PIPE)]:
            raise SystemExit(f"set_clipboard success used an unexpected subprocess seam: {clipboard_calls}")
        if successful_clipboard_process is None or successful_clipboard_process.input_bytes != b"mock clipboard text":
            raise SystemExit("set_clipboard success did not send the exact mocked clipboard bytes.")
        assert_system_handoff(
            successful_clipboard,
            label="set_clipboard success",
            source="set_clipboard",
            status="ok",
            state_changed=True,
            changed=["clipboard"],
        )

        clipboard_calls.clear()
        try:
            system.subprocess.Popen = lambda command, *, stdin: FakeClipboardProcess(  # type: ignore[assignment]
                command, stdin=stdin, returncode=1
            )
            failed_clipboard_set = system.set_clipboard({"text": "mock clipboard text"})
        finally:
            system.subprocess.Popen = original_popen  # type: ignore[assignment]
        if failed_clipboard_set.ok or clipboard_calls != [(["pbcopy"], system.subprocess.PIPE)]:
            raise SystemExit(f"set_clipboard failure did not stay on the mocked subprocess seam: {failed_clipboard_set.metadata}")
        assert_system_handoff(failed_clipboard_set, label="set_clipboard failure", source="set_clipboard", status="error")

        bad_volume = system.volume({"level": "loud"})
        if bad_volume.ok or bad_volume.metadata.get("writes_files") or bad_volume.metadata.get("controls_computer") or bad_volume.metadata.get("queues_approval"):
            raise SystemExit("volume bad level refusal missed safety metadata.")
        if bad_volume.metadata.get("raw_level") != "loud" or bad_volume.metadata.get("level") is not None:
            raise SystemExit(f"volume bad level should preserve bounded raw level metadata: {bad_volume.metadata}")
        assert_operator_limits(bad_volume.metadata, bad_volume.output, "volume bad level")
        assert_system_handoff(bad_volume, label="volume bad level", source="volume", status="refused", reason="invalid_level")

        long_bad_volume = system.volume({"level": "v" * 200})
        if long_bad_volume.ok or long_bad_volume.metadata.get("raw_level") != ("v" * 79 + "…"):
            raise SystemExit(f"volume bad level should bound raw level metadata: {long_bad_volume.metadata}")
        assert_operator_limits(long_bad_volume.metadata, long_bad_volume.output, "volume long bad level")
        assert_system_handoff(long_bad_volume, label="volume long bad level", source="volume", status="refused", reason="invalid_level")

        path_bad_volume = system.volume({"level": "/\x55sers/example/private/audio-level"})
        if path_bad_volume.ok or path_bad_volume.metadata.get("raw_level") != "<local-path>":
            raise SystemExit(f"volume bad level should redact path-shaped raw level metadata: {path_bad_volume.metadata}")
        assert_operator_limits(path_bad_volume.metadata, path_bad_volume.output, "volume path bad level")
        assert_system_handoff(path_bad_volume, label="volume path bad level", source="volume", status="refused", reason="invalid_level")

        class FakeVolumeFailure:
            returncode = 1
            stdout = ""
            stderr = "0:13 syntax /\x55sers/example/private/audio"

        try:
            system._run = lambda _args: FakeVolumeFailure()  # type: ignore[assignment]
            unavailable_volume = system.volume({})
            failed_volume_set = system.volume({"level": "42"})
        finally:
            system._run = original_run  # type: ignore[assignment]
        if not unavailable_volume.ok or "Current volume is unavailable" not in unavailable_volume.output:
            raise SystemExit(f"volume read failure should return unavailable notice: {unavailable_volume.output}")
        if "/\x55sers/operator" in unavailable_volume.output or "syntax" in unavailable_volume.output:
            raise SystemExit(f"volume read failure leaked raw stderr: {unavailable_volume.output}")
        if unavailable_volume.metadata.get("error_type") != "volume_read_unavailable" or unavailable_volume.metadata.get("error_detail_chars", 0) <= 0:
            raise SystemExit(f"volume read failure missed diagnostic metadata: {unavailable_volume.metadata}")
        assert_recovery_guidance(
            unavailable_volume.output,
            "volume read failure",
            ("System Settings", "Sound", "Privacy & Security", "setup check", "retry"),
        )
        assert_operator_limits(unavailable_volume.metadata, unavailable_volume.output, "volume read failure")
        assert_system_handoff(unavailable_volume, label="volume read failure", source="volume", status="empty")
        if failed_volume_set.ok or "Could not set volume." not in failed_volume_set.output:
            raise SystemExit(f"volume set failure should return friendly output: {failed_volume_set.output}")
        if "/\x55sers/operator" in failed_volume_set.output or "syntax" in failed_volume_set.output:
            raise SystemExit(f"volume set failure leaked raw stderr: {failed_volume_set.output}")
        if failed_volume_set.metadata.get("returncode") != 1 or failed_volume_set.metadata.get("level") != 42:
            raise SystemExit(f"volume set failure missed diagnostic metadata: {failed_volume_set.metadata}")
        assert_recovery_guidance(
            failed_volume_set.output,
            "volume set failure",
            ("System Settings", "Sound", "Privacy & Security", "setup check", "retry"),
        )
        assert_operator_limits(failed_volume_set.metadata, failed_volume_set.output, "volume set failure")
        assert_system_handoff(failed_volume_set, label="volume set failure", source="volume", status="error")

        successful_system_calls: list[list[str]] = []

        class FakeSystemSuccess:
            returncode = 0
            stdout = ""
            stderr = ""

        class FakeVolumeReadSuccess:
            returncode = 0
            stdout = "37\n"
            stderr = ""

        try:
            def successful_volume_read_run(command):
                successful_system_calls.append(command)
                return FakeVolumeReadSuccess()

            system._run = successful_volume_read_run  # type: ignore[assignment]
            successful_volume_read = system.volume({})
        finally:
            system._run = original_run  # type: ignore[assignment]
        if successful_system_calls != [["osascript", "-e", "output volume of (get volume settings)"]]:
            raise SystemExit(f"volume read success used an unexpected subprocess seam: {successful_system_calls}")
        assert_system_handoff(successful_volume_read, label="volume read success", source="volume", status="ok")

        successful_system_calls.clear()
        try:
            def successful_system_run(command):
                successful_system_calls.append(command)
                return FakeSystemSuccess()

            system._run = successful_system_run  # type: ignore[assignment]
            successful_volume_set = system.volume({"level": "42"})
        finally:
            system._run = original_run  # type: ignore[assignment]
        if successful_system_calls != [["osascript", "-e", "set volume output volume 42"]]:
            raise SystemExit(f"volume success used an unexpected subprocess seam: {successful_system_calls}")
        assert_system_handoff(
            successful_volume_set,
            label="volume set success",
            source="volume",
            status="ok",
            state_changed=True,
            changed=["output_volume"],
        )

        stop = system.stop_jarvis({})
        if not stop.ok:
            raise SystemExit("stop_jarvis should always acknowledge safely.")
        for key in ("calls_model", "executes_tools", "queues_approval", "controls_computer", "reads_private_data", "writes_files"):
            if stop.metadata.get(key):
                raise SystemExit("stop_jarvis should be non-destructive and read-only.")
        assert_operator_limits(stop.metadata, stop.output, "stop_jarvis")

        no_app = system.open_application({"name": ""})
        if no_app.ok or not no_app.metadata.get("executes_side_effect") or no_app.metadata.get("queues_approval"):
            raise SystemExit("open_application no-name refusal missed side-effect metadata.")
        assert_operator_limits(no_app.metadata, no_app.output, "open_application no-name")
        assert_system_handoff(no_app, label="open_application no-name", source="open_application", status="refused", reason="missing_app_name")

        successful_system_calls.clear()
        try:
            system._run = successful_system_run  # type: ignore[assignment]
            successful_open = system.open_application({"name": "FakeApp"})
        finally:
            system._run = original_run  # type: ignore[assignment]
        if successful_system_calls != [["open", "-a", "FakeApp"]]:
            raise SystemExit(f"open_application success used an unexpected subprocess seam: {successful_system_calls}")
        assert_system_handoff(
            successful_open,
            label="open_application success",
            source="open_application",
            status="ok",
            state_changed=True,
            changed=["application_launch"],
        )

        def raise_unexpected_open(*_args, **_kwargs):
            raise AssertionError("open_application should reject path-shaped app names before subprocess execution")

        try:
            system._run = raise_unexpected_open  # type: ignore[assignment]
            path_app = system.open_application({"name": "/\x55sers/example/private/FakeApp.app"})
        finally:
            system._run = original_run  # type: ignore[assignment]
        if path_app.ok or path_app.metadata.get("reason") != "invalid_app_name":
            raise SystemExit(f"open_application should reject path-shaped app names: {path_app.metadata}")
        if path_app.metadata.get("app") != "<local-path>" or path_app.metadata.get("executes_side_effect"):
            raise SystemExit(f"open_application path refusal missed redacted inert metadata: {path_app.metadata}")
        if "/\x55sers/" in path_app.output or "/private/" in path_app.output or "/\x55sers/" in str(path_app.metadata) or "/private/" in str(path_app.metadata):
            raise SystemExit(f"open_application path refusal leaked a local path: {path_app.output} {path_app.metadata}")
        if path_app.metadata.get("queues_approval") or path_app.metadata.get("controls_computer") or path_app.metadata.get("writes_files"):
            raise SystemExit(f"open_application path refusal should not queue approval, control computer, or write files: {path_app.metadata}")
        assert_operator_limits(path_app.metadata, path_app.output, "open_application path-shaped name")
        assert_system_handoff(path_app, label="open_application path-shaped name", source="open_application", status="refused", reason="invalid_app_name")

        class FakeAppFailure:
            returncode = 1
            stdout = ""
            stderr = "System Events denied /\x55sers/example/private/window"

        def assert_system_recovery_guidance(output: str, label: str) -> None:
            for fragment in ("System Settings", "Privacy & Security", "Accessibility", "setup check", "retry"):
                if fragment not in output:
                    raise SystemExit(f"{label} missed actionable recovery guidance {fragment!r}: {output}")

        try:
            system._run = lambda _args: FakeAppFailure()  # type: ignore[assignment]
            failed_frontmost = system.frontmost_app({})
            failed_open = system.open_application({"name": "FakeApp"})
        finally:
            system._run = original_run  # type: ignore[assignment]
        if failed_frontmost.ok or "Could not determine frontmost app." not in failed_frontmost.output:
            raise SystemExit(f"frontmost_app failure should return friendly output: {failed_frontmost.output}")
        if "/\x55sers/operator" in failed_frontmost.output or "System Events denied" in failed_frontmost.output:
            raise SystemExit(f"frontmost_app failure leaked raw stderr: {failed_frontmost.output}")
        if failed_frontmost.metadata.get("returncode") != 1 or failed_frontmost.metadata.get("reads_private_data") is not True:
            raise SystemExit(f"frontmost_app failure missed diagnostic metadata: {failed_frontmost.metadata}")
        assert_system_recovery_guidance(failed_frontmost.output, "frontmost_app failure")
        assert_operator_limits(failed_frontmost.metadata, failed_frontmost.output, "frontmost_app failure")
        if failed_open.ok or "Could not open FakeApp." not in failed_open.output:
            raise SystemExit(f"open_application failure should return friendly output: {failed_open.output}")
        if "/\x55sers/operator" in failed_open.output or "System Events denied" in failed_open.output:
            raise SystemExit(f"open_application failure leaked raw stderr: {failed_open.output}")
        if failed_open.metadata.get("returncode") != 1 or failed_open.metadata.get("executes_side_effect") is not True:
            raise SystemExit(f"open_application failure missed diagnostic metadata: {failed_open.metadata}")
        assert_system_recovery_guidance(failed_open.output, "open_application failure")
        assert_operator_limits(failed_open.metadata, failed_open.output, "open_application failure")
        assert_system_handoff(failed_open, label="open_application failure", source="open_application", status="error")

        try:
            def raise_app_failure(*_args, **_kwargs):
                raise RuntimeError("osascript crashed near /\x55sers/example/private/window")

            system._run = raise_app_failure  # type: ignore[assignment]
            exception_frontmost = system.frontmost_app({})
            exception_open = system.open_application({"name": "FakeApp"})
        finally:
            system._run = original_run  # type: ignore[assignment]
        if exception_frontmost.ok or "Could not determine frontmost app." not in exception_frontmost.output:
            raise SystemExit(f"frontmost_app exception should return friendly output: {exception_frontmost.output}")
        if "/\x55sers/operator" in exception_frontmost.output or "crashed near" in exception_frontmost.output:
            raise SystemExit(f"frontmost_app exception leaked raw text: {exception_frontmost.output}")
        if exception_frontmost.metadata.get("exception_type") != "RuntimeError":
            raise SystemExit(f"frontmost_app exception missed diagnostic metadata: {exception_frontmost.metadata}")
        assert_system_recovery_guidance(exception_frontmost.output, "frontmost_app exception")
        assert_operator_limits(exception_frontmost.metadata, exception_frontmost.output, "frontmost_app exception")
        if exception_open.ok or "Could not open FakeApp." not in exception_open.output:
            raise SystemExit(f"open_application exception should return friendly output: {exception_open.output}")
        if "/\x55sers/operator" in exception_open.output or "crashed near" in exception_open.output:
            raise SystemExit(f"open_application exception leaked raw text: {exception_open.output}")
        if exception_open.metadata.get("exception_type") != "RuntimeError":
            raise SystemExit(f"open_application exception missed diagnostic metadata: {exception_open.metadata}")
        assert_system_recovery_guidance(exception_open.output, "open_application exception")
        assert_operator_limits(exception_open.metadata, exception_open.output, "open_application exception")
        assert_system_handoff(exception_open, label="open_application exception", source="open_application", status="error")

        info = system.system_info({})
        if not info.ok or info.metadata.get("queues_approval") or info.metadata.get("writes_files"):
            raise SystemExit("system_info missed read-only safety metadata.")
        assert_operator_limits(info.metadata, info.output, "system_info direct")

        direct_calc = utilities.calculate({"expression": "12 * (4 + 3)"})
        if not direct_calc.ok or direct_calc.metadata.get("expression_chars") != len("12 * (4 + 3)"):
            raise SystemExit("calculate missed expression metadata.")
        if direct_calc.metadata.get("executes_tools") or direct_calc.metadata.get("writes_files"):
            raise SystemExit("calculate should remain read-only.")
        assert_utility_handoff(direct_calc, label="calculate success", source="calculate", status="ok")
        natural_calcs = {
            "84 divided by 3": "84 / 3 = 28",
            "12 times 8": "12 * 8 = 96",
            "15 plus 27": "15 + 27 = 42",
            "20 minus 9": "20 - 9 = 11",
            "18% of 240": "(18/100) * 240 = 43.2",
            "20 percent of 150": "(20/100) * 150 = 30",
            "sqrt(144)": "sqrt(144) = 12",
            # End-to-end value check for the silent 3+ term "and" bug: these
            # exercise the FULL path (planner phrase -> substituted expression
            # -> eval) via the already-substituted expression the planner would
            # produce, confirming the actual NUMBER is correct (30, not the old
            # silently-wrong 15 from "5 + 10 and 15" evaluating as Python `and`).
            "5 plus 10 plus 15": "5 + 10 + 15 = 30",
            "2 multiplied by 3 multiplied by 4": "2 * 3 * 4 = 24",
            # End-to-end value checks for the new sum/max/min/difference routes.
            "sum([1, 2, 3, 4])": "sum([1, 2, 3, 4]) = 10",
            "max(3, 7, 2)": "max(3, 7, 2) = 7",
            "min(5, 1, 9)": "min(5, 1, 9) = 1",
            "abs(100 - 75)": "abs(100 - 75) = 25",
        }
        for expression, expected_output in natural_calcs.items():
            natural_calc = utilities.calculate({"expression": expression})
            if not natural_calc.ok or natural_calc.output != expected_output:
                raise SystemExit(f"calculate natural expression failed for {expression!r}: {natural_calc.output}")
            if natural_calc.metadata.get("executes_tools") or natural_calc.metadata.get("writes_files") or natural_calc.metadata.get("queues_approval"):
                raise SystemExit(f"calculate natural expression missed read-only metadata: {natural_calc.metadata}")
            assert_utility_handoff(natural_calc, label=f"calculate natural {expression}", source="calculate", status="ok")
        direct_spell = utilities.spell_word({"word": "restaurant"})
        if not direct_spell.ok or direct_spell.output != "restaurant: r e s t a u r a n t":
            raise SystemExit(f"spell_word output wrong: {direct_spell.output}")
        if direct_spell.metadata.get("spell_text") != "restaurant" or direct_spell.metadata.get("letter_count") != len("restaurant"):
            raise SystemExit(f"spell_word metadata wrong: {direct_spell.metadata}")
        assert_utility_handoff(direct_spell, label="spell_word success", source="spell_word", status="ok")
        empty_spell = utilities.spell_word({"word": ""})
        if empty_spell.ok:
            raise SystemExit("spell_word should refuse missing text")
        assert_utility_handoff(empty_spell, label="spell_word missing", source="spell_word", status="refused", reason="missing_text")
        path_spell = utilities.spell_word({"word": "/\x55sers/example/private/name"})
        if path_spell.ok or "/\x55sers/" in path_spell.output or "/\x55sers/" in str(path_spell.metadata):
            raise SystemExit(f"spell_word should refuse and redact path-shaped text: {path_spell.output} {path_spell.metadata}")
        assert_utility_handoff(path_spell, label="spell_word path", source="spell_word", status="refused", reason="local_path_text")
        direct_count = utilities.count_text({"text": "hello world"})
        if not direct_count.ok or direct_count.output != "2 words, 11 characters":
            raise SystemExit(f"count_text output wrong: {direct_count.output}")
        if direct_count.metadata.get("word_count") != 2 or direct_count.metadata.get("char_count") != 11:
            raise SystemExit(f"count_text metadata wrong: {direct_count.metadata}")
        assert_utility_handoff(direct_count, label="count_text success", source="count_text", status="ok")
        empty_count = utilities.count_text({"text": ""})
        if empty_count.ok:
            raise SystemExit("count_text should refuse missing text")
        assert_utility_handoff(empty_count, label="count_text missing", source="count_text", status="refused", reason="missing_text")
        path_count = utilities.count_text({"text": "/\x55sers/example/private/name"})
        if path_count.ok or "/\x55sers/" in path_count.output or "/\x55sers/" in str(path_count.metadata):
            raise SystemExit(f"count_text should refuse and redact path-shaped text: {path_count.output} {path_count.metadata}")
        assert_utility_handoff(path_count, label="count_text path", source="count_text", status="refused", reason="local_path_text")
        korean_count = utilities.count_text({"text": "안녕하세요 세계"})
        if not korean_count.ok or korean_count.output != "2 words, 8 characters":
            raise SystemExit(f"count_text should count Korean words: {korean_count.output}")
        assert_utility_handoff(korean_count, label="count_text Korean", source="count_text", status="ok")
        accented_count = utilities.count_text({"text": "café déjà vu"})
        if not accented_count.ok or accented_count.output != "3 words, 12 characters":
            raise SystemExit(f"count_text should count accented Latin words: {accented_count.output}")
        assert_utility_handoff(accented_count, label="count_text accented", source="count_text", status="ok")
        direct_transform = utilities.transform_text({"text": "hello world", "mode": "uppercase"})
        if not direct_transform.ok or direct_transform.output != "HELLO WORLD":
            raise SystemExit(f"transform_text output wrong: {direct_transform.output}")
        if direct_transform.metadata.get("transform_mode") != "uppercase" or direct_transform.metadata.get("transform_text") != "hello world":
            raise SystemExit(f"transform_text metadata wrong: {direct_transform.metadata}")
        assert_utility_handoff(direct_transform, label="transform_text success", source="transform_text", status="ok")
        title_transform = utilities.transform_text({"text": "hello world", "mode": "title case"})
        if not title_transform.ok or title_transform.output != "Hello World":
            raise SystemExit(f"transform_text title case wrong: {title_transform.output}")
        assert_utility_handoff(title_transform, label="transform_text title", source="transform_text", status="ok")
        sentence_transform = utilities.transform_text({"text": "hELLO WORLD", "mode": "sentence case"})
        if not sentence_transform.ok or sentence_transform.output != "Hello world":
            raise SystemExit(f"transform_text sentence case wrong: {sentence_transform.output}")
        assert_utility_handoff(sentence_transform, label="transform_text sentence", source="transform_text", status="ok")
        swap_transform = utilities.transform_text({"text": "Hello World", "mode": "swap case"})
        if not swap_transform.ok or swap_transform.output != "hELLO wORLD":
            raise SystemExit(f"transform_text swap case wrong: {swap_transform.output}")
        assert_utility_handoff(swap_transform, label="transform_text swap", source="transform_text", status="ok")
        camel_transform = utilities.transform_text({"text": "hello jarvis world", "mode": "camel case"})
        if not camel_transform.ok or camel_transform.output != "helloJarvisWorld":
            raise SystemExit(f"transform_text camel case wrong: {camel_transform.output}")
        assert_utility_handoff(camel_transform, label="transform_text camel", source="transform_text", status="ok")
        pascal_transform = utilities.transform_text({"text": "hello jarvis world", "mode": "pascal case"})
        if not pascal_transform.ok or pascal_transform.output != "HelloJarvisWorld":
            raise SystemExit(f"transform_text pascal case wrong: {pascal_transform.output}")
        assert_utility_handoff(pascal_transform, label="transform_text pascal", source="transform_text", status="ok")
        snake_transform = utilities.transform_text({"text": "hello jarvis world", "mode": "snake case"})
        if not snake_transform.ok or snake_transform.output != "hello_jarvis_world":
            raise SystemExit(f"transform_text snake case wrong: {snake_transform.output}")
        assert_utility_handoff(snake_transform, label="transform_text snake", source="transform_text", status="ok")
        korean_snake_transform = utilities.transform_text({"text": "자비스 프로젝트", "mode": "snake case"})
        if not korean_snake_transform.ok or korean_snake_transform.output != "자비스_프로젝트":
            raise SystemExit(f"transform_text should preserve Korean words: {korean_snake_transform.output}")
        if korean_snake_transform.metadata.get("transform_word_count") != 2:
            raise SystemExit(f"transform_text Korean word count wrong: {korean_snake_transform.metadata}")
        assert_utility_handoff(korean_snake_transform, label="transform_text Korean snake", source="transform_text", status="ok")
        kebab_transform = utilities.transform_text({"text": "hello jarvis world", "mode": "kebab case"})
        if not kebab_transform.ok or kebab_transform.output != "hello-jarvis-world":
            raise SystemExit(f"transform_text kebab case wrong: {kebab_transform.output}")
        assert_utility_handoff(kebab_transform, label="transform_text kebab", source="transform_text", status="ok")
        initials_transform = utilities.transform_text({"text": "Sample User", "mode": "initials"})
        if not initials_transform.ok or initials_transform.output != "SU":
            raise SystemExit(f"transform_text initials wrong: {initials_transform.output}")
        assert_utility_handoff(initials_transform, label="transform_text initials", source="transform_text", status="ok")
        slug_transform = utilities.transform_text({"text": "Hello, Jarvis World!", "mode": "slugify"})
        if not slug_transform.ok or slug_transform.output != "hello-jarvis-world":
            raise SystemExit(f"transform_text slugify wrong: {slug_transform.output}")
        if slug_transform.metadata.get("transform_mode") != "slugify" or slug_transform.metadata.get("transform_word_count") != 3:
            raise SystemExit(f"transform_text slugify metadata wrong: {slug_transform.metadata}")
        assert_utility_handoff(slug_transform, label="transform_text slugify", source="transform_text", status="ok")
        no_spaces_transform = utilities.transform_text({"text": "hello world again", "mode": "remove spaces"})
        if not no_spaces_transform.ok or no_spaces_transform.output != "helloworldagain":
            raise SystemExit(f"transform_text remove spaces wrong: {no_spaces_transform.output}")
        assert_utility_handoff(no_spaces_transform, label="transform_text remove spaces", source="transform_text", status="ok")
        normalize_spaces_transform = utilities.transform_text({"text": "  hello   jarvis   world  ", "mode": "normalize spaces"})
        if not normalize_spaces_transform.ok or normalize_spaces_transform.output != "hello jarvis world":
            raise SystemExit(f"transform_text normalize spaces wrong: {normalize_spaces_transform.output}")
        assert_utility_handoff(normalize_spaces_transform, label="transform_text normalize spaces", source="transform_text", status="ok")
        sort_transform = utilities.transform_text({"text": "banana apple cherry", "mode": "sort words"})
        if not sort_transform.ok or sort_transform.output != "apple banana cherry":
            raise SystemExit(f"transform_text sort words wrong: {sort_transform.output}")
        assert_utility_handoff(sort_transform, label="transform_text sort words", source="transform_text", status="ok")
        reverse_transform = utilities.transform_text({"text": "hello world", "mode": "reverse"})
        if not reverse_transform.ok or reverse_transform.output != "dlrow olleh":
            raise SystemExit(f"transform_text reverse wrong: {reverse_transform.output}")
        assert_utility_handoff(reverse_transform, label="transform_text reverse", source="transform_text", status="ok")
        reverse_words_transform = utilities.transform_text({"text": "hello world again", "mode": "reverse words"})
        if not reverse_words_transform.ok or reverse_words_transform.output != "again world hello":
            raise SystemExit(f"transform_text reverse words wrong: {reverse_words_transform.output}")
        assert_utility_handoff(reverse_words_transform, label="transform_text reverse words", source="transform_text", status="ok")
        unique_transform = utilities.transform_text({"text": "banana apple banana", "mode": "unique words"})
        if not unique_transform.ok or unique_transform.output != "banana apple":
            raise SystemExit(f"transform_text unique words wrong: {unique_transform.output}")
        assert_utility_handoff(unique_transform, label="transform_text unique words", source="transform_text", status="ok")
        punctuation_transform = utilities.transform_text({"text": "hello, world!", "mode": "remove punctuation"})
        if not punctuation_transform.ok or punctuation_transform.output != "hello world":
            raise SystemExit(f"transform_text remove punctuation wrong: {punctuation_transform.output}")
        assert_utility_handoff(punctuation_transform, label="transform_text remove punctuation", source="transform_text", status="ok")
        repeat_transform = utilities.transform_text({"text": "hello", "mode": "repeat", "count": 3})
        if not repeat_transform.ok or repeat_transform.output != "hello hello hello" or repeat_transform.metadata.get("repeat_count") != 3:
            raise SystemExit(f"transform_text repeat wrong: {repeat_transform.output} {repeat_transform.metadata}")
        assert_utility_handoff(repeat_transform, label="transform_text repeat", source="transform_text", status="ok")
        repeat_bad_count = utilities.transform_text({"text": "hello", "mode": "repeat", "count": 21})
        if repeat_bad_count.ok:
            raise SystemExit("transform_text should refuse repeat counts outside range")
        assert_utility_handoff(repeat_bad_count, label="transform_text repeat count", source="transform_text", status="refused", reason="count_out_of_range")
        empty_transform = utilities.transform_text({"text": "", "mode": "uppercase"})
        if empty_transform.ok:
            raise SystemExit("transform_text should refuse missing text")
        assert_utility_handoff(empty_transform, label="transform_text missing", source="transform_text", status="refused", reason="missing_text")
        bad_transform = utilities.transform_text({"text": "hello", "mode": "rot13"})
        if bad_transform.ok:
            raise SystemExit("transform_text should refuse unsupported mode")
        assert_utility_handoff(bad_transform, label="transform_text unsupported", source="transform_text", status="refused", reason="unsupported_mode")
        path_transform = utilities.transform_text({"text": "/\x55sers/example/private/name", "mode": "uppercase"})
        if path_transform.ok or "/\x55sers/" in path_transform.output or "/\x55sers/" in str(path_transform.metadata):
            raise SystemExit(f"transform_text should refuse and redact path-shaped text: {path_transform.output} {path_transform.metadata}")
        assert_utility_handoff(path_transform, label="transform_text path", source="transform_text", status="refused", reason="local_path_text")
        direct_bmi = utilities.calculate_bmi({"weight": 70, "weight_unit": "kg", "height": 180, "height_unit": "cm"})
        if not direct_bmi.ok or "BMI: 21.6049" not in direct_bmi.output or "not a medical diagnosis" not in direct_bmi.output:
            raise SystemExit(f"calculate_bmi output wrong: {direct_bmi.output}")
        if direct_bmi.metadata.get("bmi_category") != "normal" or direct_bmi.metadata.get("read_only") is False:
            raise SystemExit(f"calculate_bmi metadata wrong: {direct_bmi.metadata}")
        assert_utility_handoff(direct_bmi, label="calculate_bmi success", source="calculate_bmi", status="ok")
        bad_bmi = utilities.calculate_bmi({"weight": 70, "weight_unit": "stone", "height": 180, "height_unit": "cm"})
        if bad_bmi.ok:
            raise SystemExit("calculate_bmi should refuse unsupported units")
        assert_utility_handoff(bad_bmi, label="calculate_bmi unsupported", source="calculate_bmi", status="refused", reason="unsupported_units")
        planner = RuleBasedPlanner()
        for query, expected_args in [
            ("tools please", {}),
            ("tool list please", {}),
            ("list tools please", {}),
            ("show tools", {}),
            ("available tools please", {}),
            ("what tools do you have please", {}),
            ("what tools does jarvis have", {}),
            ("what are your tools please", {}),
            ("show me your tools", {}),
            ("show me jarvis tools", {}),
            ("list jarvis tools", {}),
            ("list tools code please", {"toolset": "code"}),
        ]:
            plan = planner.plan(query)
            actions = plan.actions
            if len(actions) != 1 or actions[0].tool_name != "list_tools" or actions[0].args != expected_args:
                raise SystemExit(f"{query!r} should route to list_tools with {expected_args}: {[(a.tool_name, a.args) for a in actions]}")
        # Real gap found live 2026-07-09: "what app is focused" fell through to
        # chat while "frontmost app" and "current app" both worked.
        for query in ["frontmost app", "current app", "what app is focused", "what's the frontmost app"]:
            actions = planner.plan(query).actions
            if [a.tool_name for a in actions] != ["frontmost_app"]:
                raise SystemExit(f"frontmost_app route missed: {query!r} -> {[a.tool_name for a in actions]}")
        for query in [
            "divide 84 by 3",
            "split 84 by 3",
            "84 divided by 3",
            "multiply 12 by 8",
            "multiply 12 and 8",
            "12 times 8",
            "15 + 20",
            "add 15 and 20",
            "subtract 9 from 20",
            "20 minus 9",
            "what is 18% of 240",
            "what is 20 percent tip on 50",
            "what is an 18% tip on 240",
            "tip on 240 at 18%",
            "total with 18% tip on 240",
            "what is the total with 18% tip on 240",
            "split bill 50 between 2",
            "split the bill 50 between 2",
            "split 50 dollars between 2 people",
            "split $50 between 2 people",
            "split 50 between 2",
            "split bill 240 3 ways",
            "split a $240 bill three ways",
            "how much each for 240 split 3 ways",
            "what percent is 60 of 240",
            "60 is what percent of 240",
            "what percentage is 60 of 240",
            "percentage change from 50 to 60",
            "what is the percent increase from 50 to 60",
            "percent decrease from 60 to 50",
            "sales tax on 100 at 8%",
            "tax on 100 at 8 percent",
            "total with 8% tax on 100",
            "20 percent of 150",
            "84 over 3",
        ]:
            if [action.tool_name for action in planner.plan(query).actions] != ["calculate"]:
                raise SystemExit(f"planner missed natural calculate route: {query!r}")
        spell_routes = {
            "spell restaurant": "restaurant",
            "spell hello please": "hello",
            "spell out ai": "ai",
            "how do you spell restaurant": "restaurant",
        }
        for query, expected_word in spell_routes.items():
            actions = planner.plan(query).actions
            if [action.tool_name for action in actions] != ["spell_word"] or actions[0].args != {"word": expected_word}:
                raise SystemExit(f"planner missed spell route: {query!r} -> {actions}")
        for query in ["spell check accomodate", "spell check", "spellcheck accomodate", "correct spelling accomodate"]:
            actions = planner.plan(query).actions
            if actions and actions[0].tool_name == "spell_word":
                raise SystemExit(f"planner should not route unsupported spell-check phrase to spell_word: {query!r} -> {actions}")
        count_routes = {
            "word count hello world": "hello world",
            "word count 안녕하세요 세계": "안녕하세요 세계",
            "word count for hello world": "hello world",
            "word count of hello world": "hello world",
            "count words hello world": "hello world",
            "count words for hello world": "hello world",
            "how many words in hello world": "hello world",
            "how many words in for loop": "for loop",
            "character count hello world": "hello world",
            "letter count restaurant": "restaurant",
            "count letters in restaurant": "restaurant",
            "how many letters in hello": "hello",
            "how many letters are in restaurant": "restaurant",
            "count characters hello world": "hello world",
        }
        for query, expected_text in count_routes.items():
            actions = planner.plan(query).actions
            if [action.tool_name for action in actions] != ["count_text"] or actions[0].args != {"text": expected_text}:
                raise SystemExit(f"planner missed count_text route: {query!r} -> {actions}")
        transform_routes = {
            "uppercase hello world": {"text": "hello world", "mode": "uppercase"},
            "make hello world uppercase": {"text": "hello world", "mode": "uppercase"},
            "convert hello world to uppercase": {"text": "hello world", "mode": "uppercase"},
            "lowercase HELLO WORLD": {"text": "HELLO WORLD", "mode": "lowercase"},
            "make HELLO lowercase": {"text": "HELLO", "mode": "lowercase"},
            "convert HELLO WORLD to lowercase": {"text": "HELLO WORLD", "mode": "lowercase"},
            "uppercase the word hello": {"text": "hello", "mode": "uppercase"},
            "uppercase the phrase hello world": {"text": "hello world", "mode": "uppercase"},
            "lowercase the text HELLO WORLD": {"text": "HELLO WORLD", "mode": "lowercase"},
            "uppercase the word of mouth": {"text": "the word of mouth", "mode": "uppercase"},
            "title case hello world": {"text": "hello world", "mode": "title case"},
            "titlecase hello world": {"text": "hello world", "mode": "title case"},
            "title case the phrase hello world": {"text": "hello world", "mode": "title case"},
            "titlecase the text hello world": {"text": "hello world", "mode": "title case"},
            "turn hello world into title case": {"text": "hello world", "mode": "title case"},
            "capitalize hello world": {"text": "hello world", "mode": "capitalize"},
            "capitalize the word hello": {"text": "hello", "mode": "capitalize"},
            "capitalize the words hello world": {"text": "hello world", "mode": "capitalize"},
            "sentence case hello world": {"text": "hello world", "mode": "sentence case"},
            "sentence case the phrase hELLO WORLD": {"text": "hELLO WORLD", "mode": "sentence case"},
            "swap case Hello World": {"text": "Hello World", "mode": "swap case"},
            "camel case hello world": {"text": "hello world", "mode": "camel case"},
            "camelcase hello world": {"text": "hello world", "mode": "camel case"},
            "camel case the phrase hello world": {"text": "hello world", "mode": "camel case"},
            "make hello world camel case": {"text": "hello world", "mode": "camel case"},
            "make hello world camelcase": {"text": "hello world", "mode": "camel case"},
            "turn hello world into camel case": {"text": "hello world", "mode": "camel case"},
            "turn hello world into camelcase": {"text": "hello world", "mode": "camel case"},
            "pascal case hello world": {"text": "hello world", "mode": "pascal case"},
            "pascalcase hello world": {"text": "hello world", "mode": "pascal case"},
            "snake case hello world": {"text": "hello world", "mode": "snake case"},
            "snake case 자비스 프로젝트": {"text": "자비스 프로젝트", "mode": "snake case"},
            "snakecase hello world": {"text": "hello world", "mode": "snake case"},
            "snake case the text hello world": {"text": "hello world", "mode": "snake case"},
            "convert hello world to snake case": {"text": "hello world", "mode": "snake case"},
            "convert hello world to snakecase": {"text": "hello world", "mode": "snake case"},
            "kebab case hello world": {"text": "hello world", "mode": "kebab case"},
            "kebabcase hello world": {"text": "hello world", "mode": "kebab case"},
            "kebab case the words hello world": {"text": "hello world", "mode": "kebab case"},
            "make the phrase hello world uppercase": {"text": "hello world", "mode": "uppercase"},
            "make the word hello uppercase": {"text": "hello", "mode": "uppercase"},
            "convert the phrase hello world to uppercase": {"text": "hello world", "mode": "uppercase"},
            "turn the text hello world into snake case": {"text": "hello world", "mode": "snake case"},
            "initials the operator": {"text": "the operator", "mode": "initials"},
            "initials for the operator": {"text": "the operator", "mode": "initials"},
            "what are the initials of the operator": {"text": "the operator", "mode": "initials"},
            "get initials for the operator": {"text": "the operator", "mode": "initials"},
            "make initials the operator": {"text": "the operator", "mode": "initials"},
            "make initials for the operator": {"text": "the operator", "mode": "initials"},
            "acronym artificial intelligence": {"text": "artificial intelligence", "mode": "initials"},
            "acronym for artificial intelligence": {"text": "artificial intelligence", "mode": "initials"},
            "make acronym for artificial intelligence": {"text": "artificial intelligence", "mode": "initials"},
            "first letters the operator": {"text": "the operator", "mode": "initials"},
            "first letters of the operator": {"text": "the operator", "mode": "initials"},
            "slugify hello world": {"text": "hello world", "mode": "slugify"},
            "slug hello world": {"text": "hello world", "mode": "slugify"},
            "make hello world slugified": {"text": "hello world", "mode": "slugify"},
            "make hello world slug": {"text": "hello world", "mode": "slugify"},
            "convert hello world to slug": {"text": "hello world", "mode": "slugify"},
            "remove spaces hello world": {"text": "hello world", "mode": "remove spaces"},
            "remove spaces from hello world": {"text": "hello world", "mode": "remove spaces"},
            "no spaces hello world": {"text": "hello world", "mode": "no spaces"},
            "normalize spaces hello   world": {"text": "hello   world", "mode": "normalize spaces"},
            "normalise spaces hello   world": {"text": "hello   world", "mode": "normalize spaces"},
            "clean spaces hello   world": {"text": "hello   world", "mode": "normalize spaces"},
            "collapse spaces hello   world": {"text": "hello   world", "mode": "normalize spaces"},
            "remove extra spaces hello   world": {"text": "hello   world", "mode": "normalize spaces"},
            "normalize spaces from hello   world": {"text": "hello   world", "mode": "normalize spaces"},
            "trim hello world": {"text": "hello world", "mode": "trim spaces"},
            "trim the text hello world": {"text": "hello world", "mode": "trim spaces"},
            "trim spaces hello world": {"text": "hello world", "mode": "trim spaces"},
            "trim whitespace hello world": {"text": "hello world", "mode": "trim spaces"},
            "strip spaces hello world": {"text": "hello world", "mode": "trim spaces"},
            "strip whitespace from hello world": {"text": "hello world", "mode": "trim spaces"},
            "sort words banana apple cherry": {"text": "banana apple cherry", "mode": "sort words"},
            "sort the words banana apple cherry": {"text": "banana apple cherry", "mode": "sort words"},
            "alphabetize banana apple cherry": {"text": "banana apple cherry", "mode": "sort words"},
            "alphabetize the words banana apple cherry": {"text": "banana apple cherry", "mode": "sort words"},
            "reverse hello world": {"text": "hello world", "mode": "reverse"},
            "reverse words hello world": {"text": "hello world", "mode": "reverse words"},
            "reverse the words hello world": {"text": "hello world", "mode": "reverse words"},
            "reverse the text hello world": {"text": "hello world", "mode": "reverse text"},
            "unique words banana apple banana": {"text": "banana apple banana", "mode": "unique words"},
            "unique the words banana apple banana": {"text": "banana apple banana", "mode": "unique words"},
            "dedupe words banana apple banana": {"text": "banana apple banana", "mode": "dedupe words"},
            "dedupe the words banana apple banana": {"text": "banana apple banana", "mode": "unique words"},
            "remove punctuation hello, world!": {"text": "hello, world", "mode": "remove punctuation"},
            "remove punctuation from hello, world!": {"text": "hello, world", "mode": "remove punctuation"},
        }
        for query, expected_args in transform_routes.items():
            actions = planner.plan(query).actions
            if [action.tool_name for action in actions] != ["transform_text"] or actions[0].args != expected_args:
                raise SystemExit(f"planner missed transform_text route: {query!r} -> {actions}")
        if [action.tool_name for action in planner.plan("what is NASA").actions] != ["wiki_summary"]:
            raise SystemExit("initials/acronym routing should not hijack generic encyclopedia questions")
        if [action.tool_name for action in planner.plan("what is camelcase").actions] == ["transform_text"]:
            raise SystemExit("compact case routing should not hijack informational questions")
        if planner.plan("strip hello world").actions:
            raise SystemExit("planner should not route bare strip without a clearer text-cleanup mode")
        if planner.plan("lowercase and remove spaces hello world").actions:
            raise SystemExit("planner should not pretend to support chained text transforms yet")
        repeat_routes = {
            "repeat hello 3": {"text": "hello", "mode": "repeat", "count": 3},
            "repeat hello x3": {"text": "hello", "mode": "repeat", "count": 3},
            "repeat hello x 3": {"text": "hello", "mode": "repeat", "count": 3},
            "repeat hello 3 times": {"text": "hello", "mode": "repeat", "count": 3},
            "repeat hello three": {"text": "hello", "mode": "repeat", "count": 3},
            "repeat hello three times": {"text": "hello", "mode": "repeat", "count": 3},
        }
        for query, expected_args in repeat_routes.items():
            actions = planner.plan(query).actions
            if [action.tool_name for action in actions] != ["transform_text"] or actions[0].args != expected_args:
                raise SystemExit(f"planner missed repeat transform route: {query!r} -> {actions}")
        if planner.plan("repeat hello").actions:
            raise SystemExit("planner should not route repeat transform without an explicit count")
        if [action.tool_name for action in planner.plan("say hello 3 times").actions] != ["speak"]:
            raise SystemExit("repeat transform routing should not hijack voice-style say phrasing")
        calc_arg_routes = {
            "add 15 and 20": "15 plus 20",
            "multiply 12 and 8": "12 multiplied by 8",
            # Real, SILENT correctness bug found live 2026-07-09: 3+ term chains
            # ("add A and B and C") were substituted via a single lazy-first-
            # group pattern that only converted the FIRST "and" to "plus",
            # leaving later "and"s as the literal word -- which eval() then
            # interpreted as Python's boolean `and` operator, producing a
            # wrong-but-plausible-looking answer with no error at all ("add 5
            # and 10 and 15" evaluated as "5 + 10 and 15" -> 15, not 30).
            "add 5 and 10 and 15": "5 plus 10 plus 15",
            "add 5 and 10 and 20": "5 plus 10 plus 20",
            "multiply 2 and 3 and 4": "2 multiplied by 3 multiplied by 4",
            "15 + 20": "15 + 20",
            "84 over 3": "84 over 3",
            # Non-regression for the 2026-07-10 weather-vs-calculate fix: this
            # legitimate word-based calc phrasing must still route to
            # calculate() after excluding weather/forecast wording from the
            # broad "what's <alnum expression with a digit>" matcher.
            "what's 100 divided by 4": "100 divided by 4",
            "20 percent of 150": "20 percent of 150",
            "what is 20 percent tip on 50": "(20/100) * 50",
            "what is an 18% tip on 240": "(18/100) * 240",
            "tip on 240 at 18%": "(18/100) * 240",
            "total with 18% tip on 240": "240 * (1 + 18/100)",
            "what is the total with 18% tip on 240": "240 * (1 + 18/100)",
            "split bill 50 between 2": "50 divided by 2",
            "split the bill 50 between 2": "50 divided by 2",
            "split 50 dollars between 2 people": "50 divided by 2",
            "split $50 between 2 people": "50 divided by 2",
            "split 50 between 2": "50 divided by 2",
            "split bill 240 3 ways": "240 divided by 3",
            "split a $240 bill three ways": "240 divided by 3",
            "how much each for 240 split 3 ways": "240 divided by 3",
            "what percent is 60 of 240": "(60 / 240) * 100",
            "60 is what percent of 240": "(60 / 240) * 100",
            "what percentage is 60 of 240": "(60 / 240) * 100",
            "percentage change from 50 to 60": "((60 - 50) / 50) * 100",
            "what is the percent increase from 50 to 60": "((60 - 50) / 50) * 100",
            "percent decrease from 60 to 50": "((60 - 50) / 60) * 100",
            "sales tax on 100 at 8%": "(8/100) * 100",
            "tax on 100 at 8 percent": "(8/100) * 100",
            "total with 8% tax on 100": "100 * (1 + 8/100)",
            "half of 80": "80 / 2",
            "double 12": "12 * 2",
            "triple 7": "7 * 3",
            "square root of 144": "sqrt(144)",
            "sqrt 144": "sqrt(144)",
            # Real bug found live 2026-07-08: "what's THE square root of 144" fell
            # through to the raw calculate() eval path (which can't parse English)
            # because the planner regex required "square root of"/"sqrt" with no
            # allowance for a leading "the" -- same class of gap as other missing
            # natural-phrasing routes found this session.
            "what's the square root of 144": "sqrt(144)",
            "the square root of 81": "sqrt(81)",
            "12 squared": "12 ** 2",
            "what is 12 squared": "12 ** 2",
            "square 12": "12 ** 2",
            "3 cubed": "3 ** 3",
            "cube 3": "3 ** 3",
            # Real bug found live 2026-07-08: "N to the power of M" had no planner
            # route at all (only "squared"/"cubed" suffixes and "square"/"cube"
            # verbs existed), so it fell through to the same raw-eval failure.
            "what's 2 to the power of 10": "2 ** 10",
            "3 to the power of 4": "3 ** 4",
            "what is 3 to the 4th power": "3 ** 4",
            "2 to the 3rd power": "2 ** 3",
            "5 factorial": "factorial(5)",
            "factorial 5": "factorial(5)",
            # Real bug found live 2026-07-09: "factorial OF N" fell through to
            # chat (which computed it correctly via model reasoning, but that's
            # not deterministic/reliable for larger inputs) -- only bare
            # "factorial N" or "N factorial" were recognized.
            "factorial of 5": "factorial(5)",
            "what is factorial of 5": "factorial(5)",
            "average of 10 20 30": "(10 + 20 + 30) / 3",
            "average 10, 20, 30": "(10 + 20 + 30) / 3",
            "mean of 10 and 20": "(10 + 20) / 2",
            # Real bug found live 2026-07-08: the Oxford-comma list format "X, Y,
            # and Z" (arguably the most natural way to say it) wasn't handled --
            # the separator group only allowed a comma OR "and", not "comma then
            # and" together. Also fixed the same "what's THE average" gap found
            # elsewhere this session.
            "average of 5, 10, and 15": "(5 + 10 + 15) / 3",
            "what's the average of 5, 10, and 15": "(5 + 10 + 15) / 3",
            # Feature gaps found live 2026-07-09, implemented in a follow-up
            # pass: "sum of", "max/min of", and "difference between" all fell
            # through to chat (model-computed, not deterministic) since
            # calculate() had no route for them despite sum/max/min already
            # being allowed builtins in its eval namespace.
            "what is the sum of 1, 2, 3, 4": "sum([1, 2, 3, 4])",
            "sum of 10 20 30": "sum([10, 20, 30])",
            "what is the max of 3, 7, 2": "max(3, 7, 2)",
            "what is the maximum of 3, 7, 2": "max(3, 7, 2)",
            "what is the minimum of 5, 1, 9": "min(5, 1, 9)",
            "min of 5 1 9": "min(5, 1, 9)",
            "what is the difference between 100 and 75": "abs(100 - 75)",
            "difference between 5 and 12": "abs(5 - 12)",
            "round 3.14159": "round(3.14159)",
            "round 3.14159 to 2 decimals": "round(3.14159, 2)",
            "round 3.14159 to two decimals": "round(3.14159, 2)",
            # Real bug found live 2026-07-08: named math constants (pi/e/tau)
            # weren't recognized as a valid `round` value, only numeric literals.
            "round pi to 4 decimals": "round(pi, 4)",
            "round pi to 4 decimal places": "round(pi, 4)",
            # Real bug found live 2026-07-08: "X decimal places" (two words) had
            # no route at all -- only "X decimals" or "X places" alone matched.
            "round 3.14159 to 2 decimal places": "round(3.14159, 2)",
            "absolute value of -5": "abs(-5)",
            "abs -5": "abs(-5)",
            "what is the absolute value of -5": "abs(-5)",
            "20% off 50": "50 * (1 - 20/100)",
            "20 percent off 50": "50 * (1 - 20/100)",
            "what is 20% off 50": "50 * (1 - 20/100)",
            "price after 20 percent discount on 50": "50 * (1 - 20/100)",
            "increase 50 by 20 percent": "50 * (1 + 20/100)",
            "decrease 50 by 20 percent": "50 * (1 - 20/100)",
        }
        for query, expected_expression in calc_arg_routes.items():
            actions = planner.plan(query).actions
            if actions[0].args != {"expression": expected_expression}:
                raise SystemExit(f"planner returned wrong calculate args: {query!r} -> {actions[0].args}")
        bmi_routes = {
            "bmi 70 kg 180 cm": {"weight": 70.0, "weight_unit": "kg", "height": 180.0, "height_unit": "cm"},
            "calculate bmi 70 kg 1.8 m": {"weight": 70.0, "weight_unit": "kg", "height": 1.8, "height_unit": "m"},
            "bmi for 154 lbs and 70 inches": {"weight": 154.0, "weight_unit": "lbs", "height": 70.0, "height_unit": "inches"},
            "bmi 154 pounds 5'10\"": {"weight": 154.0, "weight_unit": "pounds", "height": 70.0, "height_unit": "inches"},
            "bmi 154 lb 5 ft 10 in": {"weight": 154.0, "weight_unit": "lb", "height": 70.0, "height_unit": "inches"},
        }
        for query, expected_args in bmi_routes.items():
            actions = planner.plan(query).actions
            if [action.tool_name for action in actions] != ["calculate_bmi"] or actions[0].args != expected_args:
                raise SystemExit(f"planner missed BMI route: {query!r} -> {actions}")
        for vague_bmi in ["bmi", "calculate bmi", "bmi 70 kg", "bmi height 180 cm"]:
            if planner.plan(vague_bmi).actions:
                raise SystemExit(f"planner should not overclaim incomplete BMI route: {vague_bmi!r}")
        unit_routes = {
            "what is 10 pounds in kg": {"value": 10.0, "from_unit": "pounds", "to_unit": "kg"},
            "what's 10 kg in pounds": {"value": 10.0, "from_unit": "kg", "to_unit": "pounds"},
            "convert 10 pounds into kilograms": {"value": 10.0, "from_unit": "pounds", "to_unit": "kilograms"},
            "how many cups in a liter": {"value": 1.0, "from_unit": "liter", "to_unit": "cups"},
            "how many cups are in one liter": {"value": 1.0, "from_unit": "liter", "to_unit": "cups"},
            "how many kg is 10 pounds": {"value": 10.0, "from_unit": "pounds", "to_unit": "kg"},
            "how many ml in 2 cups": {"value": 2.0, "from_unit": "cups", "to_unit": "ml"},
            "how many celsius is 32 fahrenheit": {"value": 32.0, "from_unit": "fahrenheit", "to_unit": "celsius"},
            # Real bug found live 2026-07-09: "how cold/hot is X in Y" fell
            # through to chat (model answered correctly but non-deterministically)
            # -- only "what is"/"what's"/"how much is" were recognized lead-ins.
            "how cold is 0 celsius in fahrenheit": {"value": 0.0, "from_unit": "celsius", "to_unit": "fahrenheit"},
            "how hot is 100 fahrenheit in celsius": {"value": 100.0, "from_unit": "fahrenheit", "to_unit": "celsius"},
            "10 pounds kg": {"value": 10.0, "from_unit": "pounds", "to_unit": "kg"},
            "10 lbs kg": {"value": 10.0, "from_unit": "lbs", "to_unit": "kg"},
            "32 fahrenheit celsius": {"value": 32.0, "from_unit": "fahrenheit", "to_unit": "celsius"},
            "10 km miles": {"value": 10.0, "from_unit": "km", "to_unit": "miles"},
            "100 cm inches": {"value": 100.0, "from_unit": "cm", "to_unit": "inches"},
            "2 cups ml": {"value": 2.0, "from_unit": "cups", "to_unit": "ml"},
            "one liter to cups": {"value": 1.0, "from_unit": "liter", "to_unit": "cups"},
            "half a mile km": {"value": 0.5, "from_unit": "mile", "to_unit": "km"},
            "half mile to km": {"value": 0.5, "from_unit": "mile", "to_unit": "km"},
            "a half mile km": {"value": 0.5, "from_unit": "mile", "to_unit": "km"},
            "quarter cup ml": {"value": 0.25, "from_unit": "cup", "to_unit": "ml"},
            "a quarter cup ml": {"value": 0.25, "from_unit": "cup", "to_unit": "ml"},
            "minus 40 fahrenheit celsius": {"value": -40.0, "from_unit": "fahrenheit", "to_unit": "celsius"},
            "negative 40 celsius fahrenheit": {"value": -40.0, "from_unit": "celsius", "to_unit": "fahrenheit"},
            "how many km is half a mile": {"value": 0.5, "from_unit": "mile", "to_unit": "km"},
            "how many ml are a quarter cup": {"value": 0.25, "from_unit": "cup", "to_unit": "ml"},
            "two cups ml": {"value": 2.0, "from_unit": "cups", "to_unit": "ml"},
            "three miles km": {"value": 3.0, "from_unit": "miles", "to_unit": "km"},
            "five pounds kg": {"value": 5.0, "from_unit": "pounds", "to_unit": "kg"},
            "ten kilometers miles": {"value": 10.0, "from_unit": "kilometers", "to_unit": "miles"},
            "twenty celsius fahrenheit": {"value": 20.0, "from_unit": "celsius", "to_unit": "fahrenheit"},
            "negative twenty celsius fahrenheit": {"value": -20.0, "from_unit": "celsius", "to_unit": "fahrenheit"},
            "minus twenty fahrenheit celsius": {"value": -20.0, "from_unit": "fahrenheit", "to_unit": "celsius"},
            "one and a half miles km": {"value": 1.5, "from_unit": "miles", "to_unit": "km"},
            "one point five miles km": {"value": 1.5, "from_unit": "miles", "to_unit": "km"},
            "three quarters cup ml": {"value": 0.75, "from_unit": "cup", "to_unit": "ml"},
            "three quarter cup ml": {"value": 0.75, "from_unit": "cup", "to_unit": "ml"},
            "how many ml are two cups": {"value": 2.0, "from_unit": "cups", "to_unit": "ml"},
            "how many km is three miles": {"value": 3.0, "from_unit": "miles", "to_unit": "km"},
            # Real gap found live 2026-07-10: "how many minutes ARE THERE in
            # 2.5 hours" swallowed "are there" into the unit-name capture
            # (to_unit="minutes are there", an unknown-unit refusal) -- only
            # "are " was an optional connector before "in", not "are there ".
            "how many minutes are there in 2.5 hours": {"value": 2.5, "from_unit": "hours", "to_unit": "minutes"},
            "how many minutes are in an hour": {"value": 1.0, "from_unit": "hour", "to_unit": "minutes"},
            "5 ft 10 in to cm": {"value": 70.0, "from_unit": "inches", "to_unit": "cm"},
            "5 feet 10 inches to cm": {"value": 70.0, "from_unit": "inches", "to_unit": "cm"},
            "five foot ten inches to cm": {"value": 70.0, "from_unit": "inches", "to_unit": "cm"},
            "five feet ten to centimeters": {"value": 70.0, "from_unit": "inches", "to_unit": "centimeters"},
            "6 ft 2 in meters": {"value": 74.0, "from_unit": "inches", "to_unit": "meters"},
            "six feet two inches in meters": {"value": 74.0, "from_unit": "inches", "to_unit": "meters"},
            "5'10\" to cm": {"value": 70.0, "from_unit": "inches", "to_unit": "cm"},
            "5' 10\" to cm": {"value": 70.0, "from_unit": "inches", "to_unit": "cm"},
            "5 ft 10\" to cm": {"value": 70.0, "from_unit": "inches", "to_unit": "cm"},
            "5'10 to cm": {"value": 70.0, "from_unit": "inches", "to_unit": "cm"},
            "height 5'10\" in cm": {"value": 70.0, "from_unit": "inches", "to_unit": "cm"},
            "convert 5'10\" to cm": {"value": 70.0, "from_unit": "inches", "to_unit": "cm"},
            "6'2\" to meters": {"value": 74.0, "from_unit": "inches", "to_unit": "meters"},
            # Real gap found live 2026-07-10: "convert 100 fahrenheit to celsius
            # and kelvin" captured the whole "celsius and kelvin" tail as one
            # garbled to_unit, producing a misleading "Cannot mix temperature
            # with other unit types" refusal instead of a clean single-target
            # conversion. The `to` capture now stops at " and ", so this
            # resolves to the first (celsius) target instead of failing.
            "convert 100 fahrenheit to celsius and kelvin": {"value": 100.0, "from_unit": "fahrenheit", "to_unit": "celsius"},
        }
        for query, expected_args in unit_routes.items():
            actions = planner.plan(query).actions
            if [action.tool_name for action in actions] != ["convert_units"] or actions[0].args != expected_args:
                raise SystemExit(f"planner missed natural unit conversion route: {query!r} -> {actions}")
        currency_actions = planner.plan("100 usd krw").actions
        if [action.tool_name for action in currency_actions] != ["convert_currency"]:
            raise SystemExit(f"terse unit route should not hijack currency conversion: {currency_actions}")
        # Real gap found live 2026-07-10: "convert 50 euros to dollars and
        # pounds" misrouted to convert_units (from_unit="euros" is not a
        # recognized physical unit, producing a confusing "Unknown unit"
        # error) instead of convert_currency, because the same " and "-tail
        # capture bug defeated the from/to currency-code check upstream.
        compound_currency_actions = planner.plan("convert 50 euros to dollars and pounds").actions
        if [action.tool_name for action in compound_currency_actions] != ["convert_currency"]:
            raise SystemExit(f"compound currency target should still route to convert_currency: {compound_currency_actions}")
        for mixed_height_guard in ["5 feet 10 pounds", "5 ft 10 in", "5'10\"", "5'10\" pounds", "5-10 to cm"]:
            if planner.plan(mixed_height_guard).actions:
                raise SystemExit(f"planner should not overclaim mixed-height conversion: {mixed_height_guard!r}")
        password_routes = {
            "generate password": {},
            "generate password 12": {"length": 12},
            "generate password 12 characters": {"length": 12},
            "generate a 16 character password": {"length": 16},
            "make a secure 24 char password": {"length": 24},
            "password 12 characters": {"length": 12},
            "password 18 chars": {"length": 18},
            "generate password 16 no symbols": {"length": 16, "include_symbols": False},
            "generate a 16 character password without symbols": {"length": 16, "include_symbols": False},
            "password 12 characters no special characters": {"length": 12, "include_symbols": False},
            "generate alphanumeric password 20": {"length": 20, "include_symbols": False},
            "generate password 20 letters and numbers": {"length": 20, "include_symbols": False},
            "make a secure password no symbols": {"include_symbols": False},
            "random password 16": {"length": 16},
            "secure password 16": {"length": 16},
            "random secure password 16 no symbols": {"length": 16, "include_symbols": False},
            "secure password 16 without symbols": {"length": 16, "include_symbols": False},
            # Real bug found live 2026-07-09: "give me a random password" and
            # similar natural request-phrasings (not "generate"/"make") fell
            # through entirely to chat, which refused ("I don't have the
            # ability...") even though generate_password is a real, working,
            # LOCAL_SAFE tool -- only "generate"/"make" verbs were recognized.
            "give me a random password": {},
            "give me a secure password": {},
            "i need a password": {},
            "create a password": {},
        }
        for query, expected_args in password_routes.items():
            actions = planner.plan(query).actions
            if [action.tool_name for action in actions] != ["generate_password"] or actions[0].args != expected_args:
                raise SystemExit(f"planner missed password route: {query!r} -> {actions}")
        for password_guard in ["password", "new password", "password status", "gmail password", "password manager", "password reset", "secure password manager"]:
            if planner.plan(password_guard).actions:
                raise SystemExit(f"planner should not overclaim account/password-info phrase: {password_guard!r}")
        uuid_routes = [
            "uuid",
            "guid",
            "generate uuid",
            "new uuid",
            "make a uuid",
            "random uuid",
            "generate guid",
            "new guid",
        ]
        for query in uuid_routes:
            actions = planner.plan(query).actions
            if [action.tool_name for action in actions] != ["generate_uuid"] or actions[0].args != {}:
                raise SystemExit(f"planner missed UUID route: {query!r} -> {actions}")
        for uuid_guard in ["what is uuid", "what is a uuid", "uuid meaning", "random id"]:
            actions = planner.plan(uuid_guard).actions
            if [action.tool_name for action in actions] == ["generate_uuid"]:
                raise SystemExit(f"planner should not generate UUIDs for explanation/random-id phrase: {uuid_guard!r}")

        oversized_calc = utilities.calculate({"expression": "1" * (utilities.MAX_EXPRESSION_CHARS + 1)})
        if oversized_calc.ok or "too large" not in oversized_calc.output:
            raise SystemExit("calculate should refuse oversized expressions.")
        if oversized_calc.metadata.get("max_chars") != utilities.MAX_EXPRESSION_CHARS or oversized_calc.metadata.get("executes_tools"):
            raise SystemExit("calculate oversized refusal missed safe metadata.")
        assert_utility_handoff(oversized_calc, label="calculate oversized", source="calculate", status="refused", reason="expression_too_large")

        dunder_calc = utilities.calculate({"expression": "__import__('os')"})
        if dunder_calc.ok or "dunder" not in dunder_calc.output:
            raise SystemExit("calculate should reject dunder/private names.")
        if dunder_calc.metadata.get("rejected_private_name") is not True or dunder_calc.metadata.get("writes_files"):
            raise SystemExit("calculate dunder refusal missed safe metadata.")
        assert_utility_handoff(dunder_calc, label="calculate dunder", source="calculate", status="refused", reason="rejected_private_name")

        leaky_calc = utilities.calculate({"expression": "open('safe-name')"})
        if leaky_calc.ok or "Could not calculate expression." not in leaky_calc.output:
            raise SystemExit(f"calculate failure should return friendly output: {leaky_calc.output}")
        if "open" in leaky_calc.output:
            raise SystemExit(f"calculate failure leaked raw evaluation text: {leaky_calc.output}")
        if leaky_calc.metadata.get("exception_type") != "NameError" or leaky_calc.metadata.get("evaluation_error") is not True:
            raise SystemExit(f"calculate failure missed diagnostic metadata: {leaky_calc.metadata}")
        if leaky_calc.metadata.get("executes_tools") or leaky_calc.metadata.get("writes_files") or leaky_calc.metadata.get("queues_approval"):
            raise SystemExit("calculate failure missed read-only safety metadata.")
        assert_utility_handoff(leaky_calc, label="calculate eval failure", source="calculate", status="error", reason="evaluation_error")
        for expression in [
            "2**20000",
            "9**9**9",
            "[0] * 100000000",
            "factorial(1000000)",
            "(lambda: 1)()",
        ]:
            bounded_calc = utilities.calculate({"expression": expression})
            if bounded_calc.ok or "Could not calculate expression." not in bounded_calc.output:
                raise SystemExit(f"calculate should refuse resource-heavy expression {expression!r}: {bounded_calc}")
            if bounded_calc.metadata.get("evaluation_error") is not True:
                raise SystemExit(f"calculate bounded refusal missed diagnostic metadata: {bounded_calc.metadata}")
            if bounded_calc.metadata.get("executes_tools") or bounded_calc.metadata.get("writes_files") or bounded_calc.metadata.get("queues_approval"):
                raise SystemExit(f"calculate bounded refusal gained authority: {bounded_calc.metadata}")
            assert_utility_handoff(
                bounded_calc,
                label=f"calculate bounded {expression}",
                source="calculate",
                status="error",
                reason="evaluation_error",
            )
        for expression in [
            "'/\x55sers/example/private/math'",
            "'/var/folders/zc/jarvis-math'",
            "'/tmp/jarvis-math'",
        ]:
            path_calc = utilities.calculate({"expression": expression})
            if path_calc.ok or path_calc.metadata.get("local_path_expression") is not True:
                raise SystemExit(f"calculate should reject path-shaped expressions before eval: {path_calc.output} {path_calc.metadata}")
            if path_calc.metadata.get("raw_expression") != "'<local-path>":
                raise SystemExit(f"calculate should redact path-shaped expression metadata: {path_calc.metadata}")
            if any(fragment in path_calc.output or fragment in str(path_calc.metadata) for fragment in ["/\x55sers/", "/var/folders/", "/tmp/"]):
                raise SystemExit(f"calculate path refusal leaked raw local path: {path_calc.output} {path_calc.metadata}")
            assert_utility_handoff(path_calc, label="calculate path refusal", source="calculate", status="refused", reason="local_path_expression")

        password = utilities.generate_password({"length": 999999})
        if not password.ok or password.metadata.get("length") != utilities.MAX_PASSWORD_LENGTH:
            raise SystemExit("generate_password should clamp huge lengths.")
        if password.metadata.get("reads_private_data") or password.metadata.get("queues_approval"):
            raise SystemExit("generate_password missed safety metadata.")
        assert_utility_handoff(password, label="generate_password success", source="generate_password", status="ok")
        no_symbol_password = utilities.generate_password({"length": 24, "include_symbols": False})
        if not no_symbol_password.ok or no_symbol_password.metadata.get("length") != 24 or no_symbol_password.metadata.get("include_symbols") is not False:
            raise SystemExit(f"generate_password no-symbol metadata wrong: {no_symbol_password.metadata}")
        if not str(no_symbol_password.output).isalnum():
            raise SystemExit(f"generate_password no-symbol output should be alphanumeric only: {no_symbol_password.output!r}")
        assert_utility_handoff(no_symbol_password, label="generate_password no symbols", source="generate_password", status="ok")
        direct_uuid = utilities.generate_uuid({})
        if not direct_uuid.ok or direct_uuid.metadata.get("uuid_version") != 4:
            raise SystemExit(f"generate_uuid metadata wrong: {direct_uuid.metadata}")
        try:
            parsed_uuid = uuid.UUID(str(direct_uuid.output), version=4)
        except ValueError as exc:
            raise SystemExit(f"generate_uuid output should be a valid UUID v4: {direct_uuid.output!r}") from exc
        if str(parsed_uuid) != str(direct_uuid.output):
            raise SystemExit(f"generate_uuid output should be canonical lowercase UUID text: {direct_uuid.output!r}")
        assert_utility_handoff(direct_uuid, label="generate_uuid success", source="generate_uuid", status="ok")
        bad_password = utilities.generate_password({"length": "bad"})
        if bad_password.ok or bad_password.metadata.get("calls_model") or bad_password.metadata.get("writes_files"):
            raise SystemExit("generate_password bad length refusal missed safety metadata.")
        if bad_password.metadata.get("raw_length") != "bad":
            raise SystemExit(f"generate_password should preserve bounded raw length metadata: {bad_password.metadata}")
        assert_utility_handoff(bad_password, label="generate_password bad length", source="generate_password", status="refused", reason="bad_length")
        bool_password = utilities.generate_password({"length": True})
        if bool_password.ok or bool_password.metadata.get("calls_model") or bool_password.metadata.get("writes_files"):
            raise SystemExit("generate_password boolean length refusal missed safety metadata.")
        if bool_password.metadata.get("length") is not None or bool_password.metadata.get("raw_length") != "True":
            raise SystemExit(f"generate_password should treat boolean length as malformed and preserve raw metadata: {bool_password.metadata}")
        assert_utility_handoff(bool_password, label="generate_password bool length", source="generate_password", status="refused", reason="bad_length")
        long_bad_password = utilities.generate_password({"length": "l" * 200})
        if long_bad_password.ok or long_bad_password.metadata.get("raw_length") != ("l" * 79 + "…"):
            raise SystemExit(f"generate_password should bound raw length metadata: {long_bad_password.metadata}")
        assert_utility_handoff(long_bad_password, label="generate_password long bad length", source="generate_password", status="refused", reason="bad_length")
        path_bad_password = utilities.generate_password({"length": "/\x55sers/example/private/password-length"})
        if path_bad_password.ok or path_bad_password.metadata.get("raw_length") != "<local-path>":
            raise SystemExit(f"generate_password leaked local path in raw length metadata: {path_bad_password.metadata}")
        assert_utility_handoff(path_bad_password, label="generate_password path length", source="generate_password", status="refused", reason="bad_length")
        temp_path_bad_password = utilities.generate_password({"length": "/var/folders/zc/password-length"})
        if temp_path_bad_password.ok or temp_path_bad_password.metadata.get("raw_length") != "<local-path>":
            raise SystemExit(f"generate_password leaked temp path in raw length metadata: {temp_path_bad_password.metadata}")
        assert_utility_handoff(temp_path_bad_password, label="generate_password temp path length", source="generate_password", status="refused", reason="bad_length")

        conversion = utilities.convert_units({"value": 10, "from_unit": "km", "to_unit": "miles"})
        if not conversion.ok or conversion.metadata.get("from_unit") != "km" or conversion.metadata.get("to_unit") != "miles":
            raise SystemExit("convert_units missed unit metadata.")
        if conversion.metadata.get("executes_tools") or conversion.metadata.get("writes_files"):
            raise SystemExit("convert_units should remain read-only.")
        assert_utility_handoff(conversion, label="convert_units success", source="convert_units", status="ok")
        # Real gap found live 2026-07-08: convert_units had length/weight/volume/
        # temperature but no time category at all -- "convert 1 hour to minutes"
        # returned "Unknown unit: hour" even though the planner correctly routed
        # to convert_units with the right args; the unit table itself was missing
        # time entries. Pin the fix with a few common time conversions.
        for value, from_unit, to_unit, expected_fragment in [
            (1, "hour", "minutes", "1 hour = 60 minutes"),
            (2, "days", "hours", "2 days = 48 hours"),
            (90, "seconds", "minutes", "90 seconds = 1.5 minutes"),
            (1, "week", "days", "1 week = 7 days"),
        ]:
            time_conversion = utilities.convert_units({"value": value, "from_unit": from_unit, "to_unit": to_unit})
            if not time_conversion.ok or expected_fragment not in time_conversion.output:
                raise SystemExit(f"convert_units time conversion wrong: {from_unit}->{to_unit}: {time_conversion.output}")
        time_length_mismatch = utilities.convert_units({"value": 1, "from_unit": "hour", "to_unit": "km"})
        if time_length_mismatch.ok or "Cannot convert time to length" not in time_length_mismatch.output:
            raise SystemExit(f"convert_units should refuse mixing time with other unit types: {time_length_mismatch.output}")
        bad_conversion = utilities.convert_units({"value": "bad", "from_unit": "x" * 200, "to_unit": "miles"})
        if bad_conversion.ok or bad_conversion.metadata.get("executes_tools") or bad_conversion.metadata.get("queues_approval"):
            raise SystemExit("convert_units bad value refusal missed safety metadata.")
        if bad_conversion.metadata.get("raw_value") != "bad":
            raise SystemExit(f"convert_units should preserve bounded raw value metadata: {bad_conversion.metadata}")
        assert_utility_handoff(bad_conversion, label="convert_units bad value", source="convert_units", status="refused", reason="bad_value")
        path_bad_conversion = utilities.convert_units({"value": "/private/tmp/jarvis-conversion-value", "from_unit": "km", "to_unit": "miles"})
        if path_bad_conversion.ok or path_bad_conversion.metadata.get("raw_value") != "<local-path>":
            raise SystemExit(f"convert_units leaked local path in raw value metadata: {path_bad_conversion.metadata}")
        assert_utility_handoff(path_bad_conversion, label="convert_units path value", source="convert_units", status="refused", reason="bad_value")
        temp_path_bad_conversion = utilities.convert_units({"value": "/tmp/jarvis-conversion-value", "from_unit": "km", "to_unit": "miles"})
        if temp_path_bad_conversion.ok or temp_path_bad_conversion.metadata.get("raw_value") != "<local-path>":
            raise SystemExit(f"convert_units leaked temp path in raw value metadata: {temp_path_bad_conversion.metadata}")
        assert_utility_handoff(temp_path_bad_conversion, label="convert_units temp path value", source="convert_units", status="refused", reason="bad_value")
        path_bad_unit = utilities.convert_units({"value": 10, "from_unit": "/var/folders/zc/jarvis-unit", "to_unit": "miles"})
        if path_bad_unit.ok or path_bad_unit.metadata.get("from_unit") != "<local-path>" or "Unknown unit: <local-path>" not in path_bad_unit.output:
            raise SystemExit(f"convert_units should redact path-shaped units: {path_bad_unit.output} {path_bad_unit.metadata}")
        if "/var/folders/" in path_bad_unit.output or "/var/folders/" in str(path_bad_unit.metadata):
            raise SystemExit(f"convert_units leaked raw path-shaped unit: {path_bad_unit.output} {path_bad_unit.metadata}")
        assert_utility_handoff(path_bad_unit, label="convert_units path unit", source="convert_units", status="refused", reason="unsupported_conversion")
        nonfinite_conversion = utilities.convert_units({"value": "nan", "from_unit": "km", "to_unit": "miles"})
        if nonfinite_conversion.ok or "finite" not in nonfinite_conversion.output:
            raise SystemExit("convert_units should reject non-finite values.")
        if nonfinite_conversion.metadata.get("executes_tools") or nonfinite_conversion.metadata.get("queues_approval"):
            raise SystemExit("convert_units non-finite refusal missed safety metadata.")
        if nonfinite_conversion.metadata.get("raw_value") != "nan":
            raise SystemExit(f"convert_units should preserve non-finite raw value metadata: {nonfinite_conversion.metadata}")
        assert_utility_handoff(nonfinite_conversion, label="convert_units nonfinite", source="convert_units", status="refused", reason="non_finite_value")


if __name__ == "__main__":
    main()
