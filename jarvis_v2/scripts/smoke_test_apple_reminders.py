"""Smoke tests for the Apple Reminders connector (mocked osascript — no app needed)."""

from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

from jarvis_v2.agent.planner import RuleBasedPlanner
from jarvis_v2.agent.types import PlannedAction, RiskLevel
from jarvis_v2.config import load_config
from jarvis_v2.tools import apple_reminders as ar
from jarvis_v2.scripts.test_runtime import approve_pending_runtime_approval, make_temp_runtime


def _tool():
    return {t.name: t for t in ar.make_apple_reminders_tools(load_config())}["apple_reminders"]


def _mock(raw: str, *, status: str | None = None):
    def framed(list_name, _limit):
        current_status = status or (ar._STATUS_FOUND if list_name else ar._STATUS_ALL)
        return (
            f"{ar._STATUS_RECORD}{ar._FIELD_SEP}{current_status}{ar._RECORD_SEP}"
            f"{raw}"
        )

    ar._run_reminders_query = framed  # type: ignore


def _reset():
    ar._run_reminders_query = None  # type: ignore


def _process_result(
    returncode: int = 0,
    *,
    stdout: str = "",
    stderr: str = "",
    **flags,
):
    return ar._BoundedProcessResult(
        returncode=returncode,
        stdout=stdout.encode(),
        stderr=stderr.encode(),
        **flags,
    )


def _database_contains(db_path: Path, needle: str) -> bool:
    with sqlite3.connect(db_path) as connection:
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
        ):
            table = str(row[0])
            columns = [
                str(column[1])
                for column in connection.execute(f'PRAGMA table_info("{table}")')
                if str(column[2]).upper() in {"TEXT", "", "JSON"}
            ]
            if not columns:
                continue
            selected = ", ".join(f'"{column}"' for column in columns)
            for values in connection.execute(f'SELECT {selected} FROM "{table}"'):
                if any(needle in str(value) for value in values if value is not None):
                    return True
    return False


def test_tool_is_personal_data_gated() -> None:
    tool = _tool()
    if tool.risk != RiskLevel.PERSONAL_DATA:
        raise SystemExit("apple_reminders should be PERSONAL_DATA")
    contract = tool.argument_contract
    if contract is None or contract.allow_unknown or [field.name for field in contract.fields] != ["list_name", "limit"]:
        raise SystemExit(f"apple_reminders argument contract is not strict: {contract}")
    limit_field = contract.fields[1]
    if limit_field.minimum != 1 or limit_field.maximum != ar.MAX_REMINDERS:
        raise SystemExit(f"apple_reminders limit contract drifted: {contract}")
    if tool.approval_argument_resolver is not ar.resolve_apple_reminders_approval:
        raise SystemExit("apple_reminders missed strict preapproval resolution")
    if tool.approval_argument_contract != contract:
        raise SystemExit("apple_reminders approved argument contract drifted")


def test_parses_and_groups_by_list() -> None:
    raw = (
        f"Groceries{ar._FIELD_SEP}Buy milk{ar._RECORD_SEP}"
        f"Groceries{ar._FIELD_SEP}Buy eggs{ar._RECORD_SEP}"
        f"Work{ar._FIELD_SEP}Send invoice{ar._RECORD_SEP}"
    )
    _mock(raw)
    try:
        out = _tool().handler({})
        if not out.ok:
            raise SystemExit(f"listing should succeed: {out.output}")
        if out.metadata.get("reminder_count") != 3 or out.metadata.get("list_count") != 2:
            raise SystemExit(f"counts wrong: {out.metadata}")
        for needle in ("Buy milk", "Buy eggs", "Send invoice", "Groceries", "Work"):
            if needle not in out.output:
                raise SystemExit(f"output missing {needle!r}: {out.output}")
    finally:
        _reset()


def test_empty_is_clean_success() -> None:
    _mock("")
    try:
        out = _tool().handler({})
        if not out.ok or out.metadata.get("reminder_count") != 0:
            raise SystemExit(f"empty should be a clean no-reminders success: {out.output} / {out.metadata}")
        if "no open reminders" not in out.output.lower():
            raise SystemExit(f"empty message should be friendly: {out.output}")
        if out.metadata.get("reminder_read_status") != "verified_empty":
            raise SystemExit(f"empty result should be explicitly verified: {out.metadata}")
        if out.metadata.get("query_verified") is not True or out.metadata.get("verified_empty") is not True:
            raise SystemExit(f"empty result missed verified-empty truth: {out.metadata}")
        if "permission" in out.output.lower() or "retry" in out.output.lower():
            raise SystemExit(f"verified empty should not be conflated with unavailable access: {out.output}")
    finally:
        _reset()


def test_lookup_failure_is_unavailable() -> None:
    def boom(_list_name, _limit):
        raise RuntimeError("PRIVATE REMINDERS ERROR DETAIL")

    ar._run_reminders_query = boom  # type: ignore
    try:
        try:
            ar.list_apple_reminders()
        except RuntimeError:
            pass
        else:
            raise SystemExit("unverified Reminders lookup should not become an empty list")
        out = _tool().handler({})
        if out.ok or out.metadata.get("reminder_read_status") != "unavailable":
            raise SystemExit(f"lookup failure should be unavailable: {out.output} / {out.metadata}")
        if "reminder_count" in out.metadata or "list_count" in out.metadata:
            raise SystemExit(f"unavailable lookup must not claim a numeric zero: {out.metadata}")
        if out.metadata.get("failure_kind") != "apple_reminders_query_failed":
            raise SystemExit(f"lookup failure missed stable failure kind: {out.metadata}")
        if out.metadata.get("query_verified") is not False or out.metadata.get("verified_empty") is not False:
            raise SystemExit(f"lookup failure should not claim verified empty: {out.metadata}")
        if out.metadata.get("failure_stage") != "query_failed" or out.metadata.get("error_type") != "RuntimeError":
            raise SystemExit(f"lookup failure missed bounded diagnostics: {out.metadata}")
        guidance = out.metadata.get("recovery_guidance")
        if not isinstance(guidance, dict) or guidance.get("version") != 1:
            raise SystemExit(f"lookup failure missed canonical recovery guidance: {out.metadata}")
        expected_truth = {
            "outcome_known": True,
            "outcome_unknown": False,
            "side_effect_possible": False,
            "retry_safe": True,
            "automatic_retry_allowed": False,
            "authorizes_retry": False,
        }
        for key, expected in expected_truth.items():
            if out.metadata.get(key) is not expected:
                raise SystemExit(f"lookup failure recovery truth drifted for {key}: {out.metadata}")
        if "PRIVATE REMINDERS ERROR DETAIL" in out.output or "PRIVATE REMINDERS ERROR DETAIL" in str(out.metadata):
            raise SystemExit(f"lookup failure leaked raw exception text: {out}")
        for expected in ("Open Reminders yourself", "Reminders access", "macOS System Settings", "setup check", "retry"):
            if expected not in out.output:
                raise SystemExit(f"unavailable result missed recovery guidance {expected!r}: {out.output}")
    finally:
        _reset()


def test_lookup_failure_kinds_are_bounded() -> None:
    failures = [
        subprocess.TimeoutExpired(["osascript"], 20),
        FileNotFoundError("PRIVATE executable path"),
    ]
    for failure in failures:
        def boom(_list_name, _limit, error=failure):
            raise error

        ar._run_reminders_query = boom  # type: ignore
        try:
            out = _tool().handler({})
            if out.ok or out.metadata.get("lookup_status") != "unavailable":
                raise SystemExit(f"{type(failure).__name__} should fail closed: {out}")
            if out.metadata.get("error_type") != type(failure).__name__:
                raise SystemExit(f"failure type should be bounded and explicit: {out.metadata}")
            if str(failure) in out.output or str(failure) in str(out.metadata):
                raise SystemExit(f"failure details leaked for {type(failure).__name__}: {out}")
        finally:
            _reset()


def test_malformed_records_fail_closed() -> None:
    # Partial protocol corruption must not be presented as a complete reminder list.
    raw = (
        f"Groceries{ar._FIELD_SEP}Buy milk{ar._RECORD_SEP}"
        f"garbage-with-no-separator{ar._RECORD_SEP}"
        f"Work{ar._FIELD_SEP}{ar._RECORD_SEP}"  # empty reminder text
    )
    _mock(raw)
    try:
        try:
            ar.list_apple_reminders()
        except ar.AppleRemindersQueryError as exc:
            if exc.stage != "malformed_output":
                raise SystemExit(f"malformed output returned wrong stage: {exc.stage}")
            pass
        else:
            raise SystemExit("malformed records should fail closed")
        out = _tool().handler({})
        if out.ok or out.metadata.get("failure_stage") != "malformed_output":
            raise SystemExit(f"malformed query output should be unavailable: {out}")
    finally:
        _reset()


def test_closed_app_failure_does_not_launch_or_retry() -> None:
    original_osascript = ar._osascript_reminders
    calls = {"query": 0}

    def closed_app(_list_name, _limit):
        calls["query"] += 1
        return _process_result(1, stderr="Application isn't running (-600)")

    ar._osascript_reminders = closed_app  # type: ignore
    try:
        try:
            ar._real_reminders_query("", ar.DEFAULT_REMINDER_LIMIT)
        except RuntimeError:
            pass
        else:
            raise SystemExit("closed Reminders app should produce an unavailable lookup")
        if calls["query"] != 1:
            raise SystemExit(f"read-only query should not retry or launch Reminders: {calls}")
    finally:
        ar._osascript_reminders = original_osascript  # type: ignore


def test_script_refuses_to_launch_closed_reminders_app() -> None:
    script = ar._REMINDERS_SCRIPT.lower()
    guard = 'if application "reminders" is not running then'
    tell = 'tell application "reminders"'
    if guard not in script or script.index(guard) >= script.index(tell):
        raise SystemExit("real Reminders query must fail before telling a closed app")
    if any(command in script for command in ("activate\n", "launch\n", "open application")):
        raise SystemExit("read-only Reminders query must not contain an app-launch command")


def test_script_requires_one_case_and_diacritic_exact_list() -> None:
    script = ar._REMINDERS_SCRIPT
    for required in (
        "considering case, diacriticals",
        "set matchedExactListCount to matchedExactListCount + 1",
        "set matchedExactList to contents of aList",
        "if matchedExactListCount is greater than 1 then",
        f'"{ar._STATUS_RECORD}{ar._FIELD_SEP}{ar._STATUS_AMBIGUOUS}{ar._RECORD_SEP}"',
    ):
        if required not in script:
            raise SystemExit(f"exact-list AppleScript boundary missed: {required}")


def test_result_limit_reports_truthful_truncation() -> None:
    raw = "".join(
        f"List{ar._FIELD_SEP}Reminder {index}{ar._RECORD_SEP}"
        for index in range(ar.MAX_REMINDERS + 1)
    )
    _mock(raw)
    try:
        out = _tool().handler({"limit": ar.MAX_REMINDERS})
        if not out.ok or out.metadata.get("reminder_count") != ar.MAX_REMINDERS:
            raise SystemExit(f"bounded reminder result count wrong: {out.metadata}")
        if out.metadata.get("truncated") is not True:
            raise SystemExit(f"bounded reminder result should report truncation: {out.metadata}")
        for expected in (
            "returned content bounded: yes",
            f"returned {ar.MAX_REMINDERS} of maximum {ar.MAX_REMINDERS}",
            "additional open items: yes",
            "Apple Event materialization bounded: no (not attested)",
        ):
            if expected not in out.output:
                raise SystemExit(f"live boundary receipt missed {expected!r}: {out.output}")
        if "Reminder 50" in out.output:
            raise SystemExit("reminder output exceeded its declared result limit")
    finally:
        _reset()


def test_runtime_marks_unavailable_lookup_unverified() -> None:
    def boom(_list_name, _limit):
        raise PermissionError("PRIVATE permission detail")

    ar._run_reminders_query = boom  # type: ignore
    try:
        with TemporaryDirectory(prefix="jarvis-apple-reminders-runtime-") as temp:
            runtime = make_temp_runtime(Path(temp))
            held = runtime.handle("apple reminders")
            approval_id = held.tool_results[0].metadata.get("approval_id")
            if type(approval_id) is not int or held.tool_results[0].metadata.get("executed_handler") is not False:
                raise SystemExit(f"personal Reminders read did not stop at approval: {held}")
            approve_pending_runtime_approval(runtime, approval_id)
            result = runtime.handle("apple reminders", approved=True, approved_approval_id=approval_id)
            if result.verified or not result.tool_results or result.tool_results[0].ok:
                raise SystemExit(f"runtime should not verify unavailable reminders: {result}")
            trace = result.metadata.get("runtime_trace", {})
            verification = next(
                (stage for stage in trace.get("stages", []) if stage.get("stage") == "verification"),
                {},
            )
            if verification.get("status") != "failed":
                raise SystemExit(f"runtime trace missed failed verification stage: {trace}")
            if "PRIVATE permission detail" in result.response or "PRIVATE permission detail" in str(trace):
                raise SystemExit("runtime leaked raw Reminders exception details")
    finally:
        _reset()


def test_local_paths_are_redacted_in_output() -> None:
    raw = (
        f"/private/tmp/secret-list{ar._FIELD_SEP}"
        f"check /\x55sers/example/secret/file.txt{ar._RECORD_SEP}"
    )
    _mock(raw)
    try:
        out = _tool().handler({})
        if "/\x55sers/example/secret" in out.output or "/private/tmp/secret-list" in out.output:
            raise SystemExit(f"local paths must be redacted: {out.output}")
        if out.output.count("<local-path>") != 2:
            raise SystemExit(f"both list names and reminder text need redaction markers: {out.output}")
    finally:
        _reset()


def test_planner_routes_apple_reminders() -> None:
    p = RuleBasedPlanner()
    for q in ["apple reminders", "what's on my apple reminders", "show my mac reminders", "reminders app"]:
        actions = [a.tool_name for a in p.plan(q).actions]
        if actions != ["apple_reminders"]:
            raise SystemExit(f"apple reminders route missed: {q!r} -> {actions}")
    # The generic Telegram reminders route must still win for plain "my reminders".
    for q in ["my reminders", "list reminders"]:
        actions = [a.tool_name for a in p.plan(q).actions]
        if actions != ["list_reminders"]:
            raise SystemExit(f"plain reminders should stay on list_reminders: {q!r} -> {actions}")

    exact_cases = {
        "show my Apple reminders from list: Jarvis V3 Proof": ("Jarvis V3 Proof", 5),
        'mac reminders in the list: "Proof List"': ("Proof List", 5),
        "show my Apple reminders from list: Jarvis V3 Proof; limit: 1": ("Jarvis V3 Proof", 1),
        "show my Apple reminders from list: Jarvis V3 Proof; limit: banana": ("Jarvis V3 Proof", "banana"),
        "애플 리마인더 목록: 자비스 V3 시험": ("자비스 V3 시험", 5),
        "내 아이폰 리마인더 리스트: 시험": ("시험", 5),
    }
    for query, (expected_name, expected_limit) in exact_cases.items():
        plan = p.plan(query)
        if [action.tool_name for action in plan.actions] != ["apple_reminders"]:
            raise SystemExit(f"bounded Apple reminders route missed: {query!r} -> {plan.actions}")
        if plan.actions[0].args != {"list_name": expected_name, "limit": expected_limit}:
            raise SystemExit(f"bounded Apple reminders args drifted: {query!r} -> {plan.actions[0].args}")


def test_query_passes_exact_filter_and_limit_without_interpolation() -> None:
    captured = []
    exact_name = 'Proof "quoted" :|| list'

    def capture(list_name, limit):
        captured.append((list_name, limit))
        return (
            f"{ar._STATUS_RECORD}{ar._FIELD_SEP}{ar._STATUS_FOUND}{ar._RECORD_SEP}"
            f"{exact_name}{ar._FIELD_SEP}Proof item{ar._RECORD_SEP}"
        )

    ar._run_reminders_query = capture  # type: ignore
    try:
        out = _tool().handler({"list_name": exact_name, "limit": 1})
        if not out.ok or captured != [(exact_name, 1)]:
            raise SystemExit(f"exact filter/limit did not reach query seam unchanged: {captured} / {out}")
        metadata = out.metadata
        if metadata.get("exact_list_filter_applied") is not True or metadata.get("fetch_limit") != 2:
            raise SystemExit(f"exact-list bounded metadata missing: {metadata}")
        if 'Proof "quoted"' in str(metadata):
            raise SystemExit(f"exact list identity leaked into metadata: {metadata}")
    finally:
        _reset()


def test_real_query_uses_absolute_binary_and_bounded_argv() -> None:
    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return _process_result()

    with patch.object(ar, "_run_bounded_process", side_effect=fake_run):
        output = ar._real_reminders_query('Proof "list"', 2)
        if output != "":
            raise SystemExit(f"empty real-query double should stay empty: {output!r}")
        command = captured.get("command") or []
        if command[:3] != [ar.OSASCRIPT_PATH, "-e", ar._REMINDERS_SCRIPT]:
            raise SystemExit(f"Reminders query did not use the absolute system executable: {command}")
        if command[-3:] != ["--", 'Proof "list"', "2"]:
            raise SystemExit(f"list/limit must be argv rather than script interpolation: {command}")
        if captured["kwargs"]:
            raise SystemExit(f"Reminders command wrapper received unexpected overrides: {captured['kwargs']}")


def test_bounded_process_stops_and_reaps_on_overflow_and_timeout() -> None:
    for command, kwargs, expected in (
        (
            [sys.executable, "-c", "import sys; sys.stdout.buffer.write(b'x'*65536); sys.stdout.flush()"],
            {"stdout_cap": 32, "stderr_cap": 32, "timeout": 2.0},
            "overflowed",
        ),
        (
            [sys.executable, "-c", "import sys; sys.stderr.buffer.write(b'x'*65536); sys.stderr.flush()"],
            {"stdout_cap": 32, "stderr_cap": 32, "timeout": 2.0},
            "overflowed",
        ),
        (
            [sys.executable, "-c", "import time; time.sleep(5)"],
            {"stdout_cap": 32, "stderr_cap": 32, "timeout": 0.05},
            "timed_out",
        ),
    ):
        result = ar._run_bounded_process(command, **kwargs)
        if getattr(result, expected) is not True or result.process_reaped is not True:
            raise SystemExit(f"bounded subprocess did not stop/reap on {expected}: {result}")
        if len(result.stdout) > kwargs["stdout_cap"] or len(result.stderr) > kwargs["stderr_cap"]:
            raise SystemExit(f"bounded subprocess retained over-cap output: {result}")


def test_bounded_process_reaps_after_selector_failure() -> None:
    process = Mock()
    process.pid = 424242
    process.poll.return_value = None
    process.wait.return_value = 0
    process.stdout = Mock()
    process.stderr = Mock()
    selector = Mock()
    selector.register.side_effect = RuntimeError("synthetic selector failure")
    with (
        patch.object(ar.subprocess, "Popen", return_value=process),
        patch.object(ar.selectors, "DefaultSelector", return_value=selector),
        patch.object(ar.os, "killpg") as killpg,
    ):
        try:
            ar._run_bounded_process(["/usr/bin/false"])
        except RuntimeError as exc:
            if "synthetic selector failure" not in str(exc):
                raise
        else:
            raise SystemExit("selector failure should propagate after cleanup")
    killpg.assert_called_once_with(424242, ar.signal.SIGTERM)
    if process.wait.call_count < 1:
        raise SystemExit("post-start exception did not reap the child process")
    process.stdout.close.assert_called_once()
    process.stderr.close.assert_called_once()


def test_script_bounds_collection_before_return() -> None:
    script = ar._REMINDERS_SCRIPT
    for required in (
        "set fetchLimit to requestedLimit + 1",
        "if fetchedCount is greater than or equal to fetchLimit then return theOutput",
        f"my boundedText(fullListName, {ar.MAX_LIST_NAME_CHARS})",
        f"my boundedText(name of aReminder as text, {ar.MAX_REMINDER_TEXT_CHARS})",
    ):
        if required not in script:
            raise SystemExit(f"AppleScript query missed collection bound: {required}")
    if "requestedListName" not in script or "fullListName is requestedListName" not in script:
        raise SystemExit("AppleScript query missed exact-list filtering")


def test_injected_delimiters_and_terminal_controls_are_data() -> None:
    encoded_list = "Proof%3A%3A%3AList%7C%7C%7C%25"
    encoded_item = "safe\u001b]8;;https://example.test\u0007link\u202e.txt%3A%3A%3A%7C%7C%7C%25"
    _mock(f"{encoded_list}{ar._FIELD_SEP}{encoded_item}{ar._RECORD_SEP}")
    try:
        out = _tool().handler({"list_name": "Proof:::List|||%", "limit": 1})
        if not out.ok:
            raise SystemExit(f"escaped delimiter record should parse: {out}")
        for forbidden in ("\u001b", "\u0007", "\u202e"):
            if forbidden in out.output:
                raise SystemExit(f"terminal/bidi control leaked into output: {out.output!r}")
        for expected in ("Proof:::List|||%", ":::|||%"):
            if expected not in out.output:
                raise SystemExit(f"escaped reminder data was not preserved: {out.output!r}")
    finally:
        _reset()


def test_invalid_arguments_fail_before_query() -> None:
    calls = []

    def unexpected(list_name, limit):
        calls.append((list_name, limit))
        return ""

    ar._run_reminders_query = unexpected  # type: ignore
    cases = [
        {"limit": 0},
        {"limit": ar.MAX_REMINDERS + 1},
        {"limit": "1"},
        {"limit": True},
        {"list_name": "bad\nlist"},
        {"list_name": "x" * (ar.MAX_LIST_NAME_CHARS + 1)},
        {"unknown": "value"},
    ]
    try:
        for args in cases:
            out = _tool().handler(args)
            if out.ok or out.metadata.get("failure_stage") != "argument_validation":
                raise SystemExit(f"invalid args did not fail before query: {args} -> {out}")
            if out.metadata.get("reads_personal_data") is not False:
                raise SystemExit(f"invalid args falsely claimed a personal read: {out.metadata}")
        if calls:
            raise SystemExit(f"invalid args reached Reminders query: {calls}")
    finally:
        _reset()


def test_stage_specific_failures_are_content_free() -> None:
    original_osascript = ar._osascript_reminders
    cases = {
        "app_not_running": "Application isn't running (-600)",
        "permission_denied": "Not authorized to send Apple events. (-1743)",
        "query_failed": "PRIVATE ACCOUNT DETAIL unknown failure",
    }
    try:
        for expected_stage, stderr in cases.items():
            ar._osascript_reminders = lambda _name, _limit, error=stderr: _process_result(  # type: ignore
                1, stderr=error
            )
            out = _tool().handler({"list_name": "Proof", "limit": 1})
            if out.ok or out.metadata.get("failure_stage") != expected_stage:
                raise SystemExit(f"failure stage classification drifted: {expected_stage} -> {out}")
            if stderr in out.output or stderr in str(out.metadata):
                raise SystemExit(f"stage-specific failure leaked stderr: {out}")
            if out.metadata.get("suppress_output_persistence") is not True:
                raise SystemExit(f"Reminders failure missed persistence suppression: {out.metadata}")
            if "persistence_summary" in out.metadata:
                raise SystemExit("connector-controlled persistence summary must not survive")
            expected_status = "not_running" if expected_stage == "app_not_running" else "unknown"
            if out.metadata.get("app_running_precheck_status") != expected_status:
                raise SystemExit(f"app-running precheck status overclaimed: {out.metadata}")
            if expected_stage == "query_failed" and (
                "app_running_precheck" in out.metadata
                or "app_running_precheck_passed" in out.metadata
            ):
                raise SystemExit(f"generic process failure claimed a passed precheck: {out.metadata}")
    finally:
        ar._osascript_reminders = original_osascript  # type: ignore


def test_app_not_running_approved_failure_is_known_consumed_and_fresh_request_only() -> None:
    calls: list[tuple[str, int]] = []

    def app_not_running(list_name, limit):
        calls.append((list_name, limit))
        raise ar.AppleRemindersQueryError("app_not_running")

    ar._run_reminders_query = app_not_running  # type: ignore
    try:
        with TemporaryDirectory(prefix="jarvis-apple-reminders-known-failure-") as temp:
            runtime = make_temp_runtime(Path(temp))
            command = "show my Apple reminders from list: Jarvis V3 Proof; limit: 1"
            held = runtime.handle(command)
            approval_id = held.tool_results[0].metadata.get("approval_id")
            if type(approval_id) is not int or calls:
                raise SystemExit(f"app-not-running fixture crossed approval early: {held} / {calls}")
            approve_pending_runtime_approval(runtime, approval_id)

            failed = runtime.handle(
                command,
                approved=True,
                approved_approval_id=approval_id,
            )
            if failed.verified or calls != [("Jarvis V3 Proof", 1)]:
                raise SystemExit(f"approved app-not-running attempt had wrong execution truth: {failed} / {calls}")
            result = failed.tool_results[0]
            expected_metadata = {
                "failure_stage": "app_not_running",
                "failure_kind": "apple_reminders_app_not_running",
                "query_verified": False,
                "outcome_known": True,
                "outcome_unknown": False,
                "execution_outcome_unknown": False,
                "side_effect_possible": False,
                "retry_safe": True,
                "automatic_retry_allowed": False,
                "authorizes_retry": False,
                "approved_execution_outcome": "failed",
                "approved_execution_outcome_reason": "exact_failure",
                "app_launch_requested": False,
                "app_running_precheck_status": "not_running",
                "app_running_precheck_performed": True,
                "app_running_precheck_passed": False,
            }
            for key, expected in expected_metadata.items():
                if result.metadata.get(key) != expected:
                    raise SystemExit(f"app-not-running approval truth drifted for {key}: {result.metadata}")
            for expected_text in (
                "approved attempt is consumed and cannot be replayed",
                "submit the same bounded read as a fresh request",
                "new one-shot approval",
            ):
                if expected_text not in failed.response:
                    raise SystemExit(f"app-not-running retry guidance missed {expected_text!r}: {failed.response}")

            claim = runtime.store.get_approval_execution_claim(approval_id)
            evidence = runtime.store.classify_approval_execution_evidence(approval_id, limit=10)
            if (
                claim is None
                or claim["outcome"] != "failed"
                or not claim["completed_at"]
                or evidence.verdict != "APPROVAL_EXECUTION_FAILED"
                or evidence.valid_execution_proof
                or evidence.outcome_unknown
            ):
                raise SystemExit(f"app-not-running durable execution evidence drifted: {claim} / {evidence}")
            proof = runtime.handle(f"approval chain proof {approval_id}")
            if (
                "Verdict: APPROVAL_EXECUTION_FAILED" not in proof.response
                or "Valid execution proof: no" not in proof.response
                or "outcome is unknown" in proof.response.lower()
            ):
                raise SystemExit(f"app-not-running approval proof was not an exact known failure: {proof.response}")

            replay = runtime.handle(
                command,
                approved=True,
                approved_approval_id=approval_id,
            )
            if calls != [("Jarvis V3 Proof", 1)] or "already been used" not in replay.response:
                raise SystemExit(f"consumed Reminders approval replayed: {replay} / {calls}")

            fresh = runtime.handle(command)
            fresh_id = fresh.tool_results[0].metadata.get("approval_id")
            if (
                type(fresh_id) is not int
                or fresh_id == approval_id
                or fresh.tool_results[0].metadata.get("requires_confirmation") is not True
                or calls != [("Jarvis V3 Proof", 1)]
            ):
                raise SystemExit(f"fresh Reminders request did not require a new approval: {fresh} / {calls}")
    finally:
        _reset()


def test_oversized_or_overcount_output_fails_closed() -> None:
    _mock("x" * (ar._MAX_QUERY_OUTPUT_CHARS + 1))
    try:
        out = _tool().handler({"limit": 1})
        if out.ok or out.metadata.get("failure_stage") != "output_too_large":
            raise SystemExit(f"oversized raw output did not fail closed: {out}")
    finally:
        _reset()

    raw = "".join(
        f"L{ar._FIELD_SEP}R{index}{ar._RECORD_SEP}" for index in range(4)
    )
    _mock(raw)
    try:
        out = _tool().handler({"limit": 1})
        if out.ok or out.metadata.get("failure_stage") != "output_too_large":
            raise SystemExit(f"more than limit+1 records did not fail closed: {out}")
    finally:
        _reset()


def test_exact_list_not_found_is_not_verified_empty() -> None:
    _mock("", status=ar._STATUS_NOT_FOUND)
    try:
        out = _tool().handler({"list_name": "Missing exact proof list", "limit": 1})
        if out.ok or out.metadata.get("failure_stage") != "list_not_found":
            raise SystemExit(f"missing exact list was presented as verified empty: {out}")
        if out.metadata.get("verified_empty") is not False or out.metadata.get("query_verified") is not False:
            raise SystemExit(f"missing exact list claimed verified data: {out.metadata}")
        if "did not substitute another list" not in out.output:
            raise SystemExit(f"missing exact list guidance lost fail-closed truth: {out.output}")
    finally:
        _reset()


def test_exact_list_identity_and_ambiguity_fail_closed() -> None:
    requested = "Jarvis V3 Proof"
    identity_cases = (
        ("Completely Different List", "wrong-list identity"),
        ("jarvis V3 Proof", "case mismatch"),
        ("Jarvis V3 Proo\N{COMBINING ACUTE ACCENT}f", "diacritic mismatch"),
    )
    try:
        for returned_name, label in identity_cases:
            _mock(f"{returned_name}{ar._FIELD_SEP}harmless dummy{ar._RECORD_SEP}")
            out = _tool().handler({"list_name": requested, "limit": 1})
            if out.ok or out.metadata.get("failure_stage") != "exact_list_mismatch":
                raise SystemExit(f"{label} crossed exact-list boundary: {out}")
            if out.metadata.get("query_verified") is not False:
                raise SystemExit(f"{label} claimed verified data: {out.metadata}")

        _mock("", status=ar._STATUS_AMBIGUOUS)
        ambiguous = _tool().handler({"list_name": requested, "limit": 1})
        if ambiguous.ok or ambiguous.metadata.get("failure_stage") != "list_ambiguous":
            raise SystemExit(f"duplicate exact list was not rejected: {ambiguous}")
        if "More than one" not in ambiguous.output or ambiguous.metadata.get("retryable") is not False:
            raise SystemExit(f"duplicate exact-list guidance was not fail-closed: {ambiguous}")
    finally:
        _reset()


def test_success_metadata_is_content_free_and_truthful() -> None:
    secret_list = "PRIVATE PROOF LIST 8842"
    secret_item = "PRIVATE REMINDER CONTENT 8842"
    _mock(f"{secret_list}{ar._FIELD_SEP}{secret_item}{ar._RECORD_SEP}")
    try:
        out = _tool().handler({"list_name": secret_list, "limit": 1})
        if not out.ok or secret_item not in out.output:
            raise SystemExit(f"current-turn display lost expected reminder data: {out}")
        if secret_list in str(out.metadata) or secret_item in str(out.metadata):
            raise SystemExit(f"private reminder data leaked into metadata: {out.metadata}")
        expected = {
            "reads_private_data": True,
            "source_scope": "shared_macos_account",
            "uses_v2_state": False,
            "v2_state_read": False,
            "returned_content_bounded": True,
            "apple_event_materialization_bounded": False,
            "result_content_in_metadata": False,
            "suppress_output_persistence": True,
            "exact_list_filter_applied": True,
            "app_launch_requested": False,
            "app_running_precheck_configured": True,
            "app_running_precheck_status": "passed",
            "app_running_precheck": True,
            "app_running_precheck_passed": True,
            "app_launch_race_eliminated": False,
        }
        for key, value in expected.items():
            if out.metadata.get(key) != value:
                raise SystemExit(f"success metadata drifted for {key}: {out.metadata}")
        if "query_bounded" in out.metadata or "persistence_summary" in out.metadata:
            raise SystemExit(f"connector retained an overclaim or connector-owned receipt: {out.metadata}")
    finally:
        _reset()


def test_runtime_persists_only_content_free_summary() -> None:
    fixed_list = "Jarvis V3 Proof"
    private_item = "PRIVATE APPLE REMINDER RESULT 9361"
    calls: list[tuple[str, int]] = []

    def query(list_name, limit):
        calls.append((list_name, limit))
        return (
            f"{ar._STATUS_RECORD}{ar._FIELD_SEP}{ar._STATUS_FOUND}{ar._RECORD_SEP}"
            f"{fixed_list}{ar._FIELD_SEP}{private_item}{ar._RECORD_SEP}"
        )

    ar._run_reminders_query = query  # type: ignore
    try:
        with TemporaryDirectory(prefix="jarvis-apple-reminders-retention-") as temp:
            runtime = make_temp_runtime(Path(temp))
            command = f"show my Apple reminders from list: {fixed_list}; limit: 1"
            held = runtime.handle(command)
            approval_id = held.tool_results[0].metadata.get("approval_id")
            if (
                type(approval_id) is not int
                or held.verified
                or held.tool_results[0].metadata.get("failure_kind") != "approval_required"
                or held.tool_results[0].metadata.get("executed_handler") is not False
                or calls
            ):
                raise SystemExit(f"unapproved Reminders read crossed its one-shot gate: {held} / {calls}")

            invalid = runtime.handle(
                f"show my Apple reminders from list: {fixed_list}; limit: banana"
            )
            if (
                invalid.verified
                or invalid.tool_results[0].metadata.get("failure_kind")
                not in {"tool_arguments_invalid", "argument_contract_invalid"}
                or invalid.tool_results[0].metadata.get("requires_confirmation") is not False
                or calls
            ):
                raise SystemExit(f"invalid Reminders args reached approval/query: {invalid} / {calls}")

            pending_before = len(runtime.store.list_pending_approvals(status="pending", limit=100))
            semantic_invalid = runtime.executor.execute(
                PlannedAction(
                    "apple_reminders",
                    {"list_name": "x" * (ar.MAX_LIST_NAME_CHARS + 1), "limit": 1},
                    "synthetic semantic-invalid preapproval proof",
                ),
                approved=False,
            )
            pending_after = len(runtime.store.list_pending_approvals(status="pending", limit=100))
            if (
                semantic_invalid.ok
                or semantic_invalid.metadata.get("failure_kind")
                != "apple_reminders_approval_arguments_invalid"
                or semantic_invalid.metadata.get("requires_confirmation") is not False
                or semantic_invalid.metadata.get("executed_handler") is not False
                or pending_after != pending_before
                or calls
            ):
                raise SystemExit(
                    f"semantic-invalid Reminders args queued/executed: {semantic_invalid} / {calls}"
                )

            transition = approve_pending_runtime_approval(runtime, approval_id)
            if (
                transition.metadata.get("rerun_user_input")
                != "show my Apple reminders from list: <private-list>; limit: 1"
                or calls
            ):
                raise SystemExit("approval review executed or changed the Reminders request")
            result = runtime.handle(command, approved=True, approved_approval_id=approval_id)
            if not result.verified or private_item not in result.response:
                raise SystemExit(f"private reminder was not returned for the current turn: {result}")
            if calls != [(fixed_list, 1)]:
                raise SystemExit(f"approved exact rerun did not query exactly once: {calls}")
            if any(
                marker in str(result.metadata.get("runtime_trace") or {})
                for marker in (fixed_list, private_item)
            ):
                raise SystemExit("private reminder request/result leaked into the runtime trace")
            if fixed_list in result.user_input or "<private-list>" not in result.user_input:
                raise SystemExit(f"runtime did not redact the exact private list request: {result.user_input!r}")
            recent_messages = runtime.store.recent_messages(limit=10)
            recent_runs = runtime.store.recent_tool_runs(limit=10)
            durable = "\n".join(
                [str(row["content"]) + str(row["metadata"]) for row in recent_messages]
                + [str(row["output"]) + str(row["metadata"]) for row in recent_runs]
            )
            if private_item in durable:
                raise SystemExit("private reminder result survived in durable message/tool audit rows")
            if private_item in str(runtime.chat.history):
                raise SystemExit("private reminder result survived in model chat history")
            expected_receipt = (
                "apple_reminders: sensitive result verified; private content was displayed "
                "transiently and not retained."
            )
            if expected_receipt not in durable:
                raise SystemExit(f"durable audit missed runtime-owned proof receipt: {durable}")
            if _database_contains(Path(temp) / "jarvis.sqlite", private_item):
                raise SystemExit("private reminder result survived elsewhere in SQLite")
    finally:
        _reset()


def test_same_session_approval_query_once_boundary_and_retention() -> None:
    fixed_list = "Jarvis V3 Proof"
    private_item = "PRIVATE SAME SESSION REMINDER 6137"
    private_extra = "PRIVATE TRUNCATED REMINDER 6137"
    calls: list[tuple[str, int]] = []

    def query(list_name, limit):
        calls.append((list_name, limit))
        return (
            f"{ar._STATUS_RECORD}{ar._FIELD_SEP}{ar._STATUS_FOUND}{ar._RECORD_SEP}"
            f"{fixed_list}{ar._FIELD_SEP}{private_item}{ar._RECORD_SEP}"
            f"{fixed_list}{ar._FIELD_SEP}{private_extra}{ar._RECORD_SEP}"
        )

    ar._run_reminders_query = query  # type: ignore
    try:
        with TemporaryDirectory(prefix="jarvis-apple-reminders-same-session-") as temp:
            root = Path(temp)
            runtime = make_temp_runtime(root)
            command = f"show my Apple reminders from list: {fixed_list}; limit: 1"
            held = runtime.handle(command)
            approval_id = held.tool_results[0].metadata.get("approval_id")
            if type(approval_id) is not int or calls:
                raise SystemExit(f"same-session request crossed approval before review: {held} / {calls}")

            pending_note = (
                runtime.vault.root_path / "Automations" / "Pending Approvals.md"
            ).read_text(encoding="utf-8")
            for expected in (
                "personal Apple Reminders read",
                "Open Reminders yourself and keep it open before approving",
                "Jarvis will not launch the app",
                "one exact list and bounded limit",
                "one-shot approval is consumed",
                "fresh bounded request instead of replaying it",
            ):
                if expected not in pending_note:
                    raise SystemExit(
                        f"Apple Reminders pending-approval mirror missed {expected!r}: {pending_note}"
                    )

            readiness = runtime.handle(f"approval readiness {approval_id}")
            packet = runtime.handle(f"approval packet {approval_id}")
            if not readiness.verified or not packet.verified or fixed_list not in packet.response:
                raise SystemExit("same-session last look did not show the exact approved list")
            for label, review in (("readiness", readiness), ("last-look", packet)):
                for expected in (
                    "personal Apple Reminders read",
                    "Open Reminders yourself and keep it open before approving",
                    "Jarvis will not launch the app",
                    "one exact list and bounded limit",
                    "one-shot approval is consumed",
                    "fresh bounded request instead of replaying it",
                ):
                    if expected not in review.response:
                        raise SystemExit(
                            f"Apple Reminders {label} missed app-open preapproval guidance "
                            f"{expected!r}: {review.response}"
                        )
            if calls:
                raise SystemExit(f"readiness/last-look unexpectedly queried Reminders: {calls}")

            approved = runtime.handle(f"approve approval {approval_id}")
            if not approved.verified or private_item not in approved.response:
                raise SystemExit(f"same-session approved read lost its transient result: {approved}")
            if private_extra in approved.response:
                raise SystemExit("same-session approved read displayed more than its approved limit")
            for expected in (
                "returned content bounded: yes",
                "exact list identity verified: yes",
                "returned 1 of maximum 1",
                "additional open items: yes",
                "Apple Event materialization bounded: no (not attested)",
            ):
                if expected not in approved.response:
                    raise SystemExit(f"same-session live boundary missed {expected!r}: {approved.response}")
            if calls != [(fixed_list, 1)]:
                raise SystemExit(f"same-session approval did not query exactly once: {calls}")

            trace = approved.metadata.get("runtime_trace") or {}
            for expected in (
                "'returned_content_bounded': True",
                "'exact_list_identity_verified': True",
                "'requested_limit': 1",
                "'returned_reminder_count': 1",
                "'truncated': True",
                "'apple_event_materialization_bounded': False",
            ):
                if expected not in str(trace):
                    raise SystemExit(f"runtime trace missed safe boundary fact {expected}: {trace}")

            apple_runs = [
                row
                for row in runtime.store.recent_tool_runs(limit=20)
                if str(row["tool_name"]) == "apple_reminders"
                and int(row["approved"]) == 1
                and int(row["approval_id"]) == approval_id
            ]
            if len(apple_runs) != 1:
                raise SystemExit(f"same-session approved read lacked one tool audit: {apple_runs}")
            audit_metadata = json.loads(str(apple_runs[0]["metadata"] or "{}"))
            expected_audit = {
                "returned_content_bounded": True,
                "exact_list_filter_applied": True,
                "exact_list_identity_verified": True,
                "requested_limit": 1,
                "returned_reminder_count": 1,
                "truncated": True,
                "apple_event_materialization_bounded": False,
                "query_verified": True,
                "private_content_retained": False,
                "content_retention": "transient_only",
            }
            for key, value in expected_audit.items():
                if audit_metadata.get(key) != value:
                    raise SystemExit(f"durable safe boundary fact drifted for {key}: {audit_metadata}")
            for secret in (private_item, private_extra):
                if _database_contains(root / "jarvis.sqlite", secret):
                    raise SystemExit(f"same-session private content survived in SQLite: {secret}")
                if secret in str(runtime.chat.history):
                    raise SystemExit(f"same-session private content survived in chat history: {secret}")
    finally:
        _reset()


def main() -> None:
    test_tool_is_personal_data_gated()
    test_parses_and_groups_by_list()
    test_empty_is_clean_success()
    test_lookup_failure_is_unavailable()
    test_lookup_failure_kinds_are_bounded()
    test_malformed_records_fail_closed()
    test_closed_app_failure_does_not_launch_or_retry()
    test_script_refuses_to_launch_closed_reminders_app()
    test_script_requires_one_case_and_diacritic_exact_list()
    test_result_limit_reports_truthful_truncation()
    test_runtime_marks_unavailable_lookup_unverified()
    test_local_paths_are_redacted_in_output()
    test_planner_routes_apple_reminders()
    test_query_passes_exact_filter_and_limit_without_interpolation()
    test_real_query_uses_absolute_binary_and_bounded_argv()
    test_bounded_process_stops_and_reaps_on_overflow_and_timeout()
    test_bounded_process_reaps_after_selector_failure()
    test_script_bounds_collection_before_return()
    test_injected_delimiters_and_terminal_controls_are_data()
    test_invalid_arguments_fail_before_query()
    test_stage_specific_failures_are_content_free()
    test_app_not_running_approved_failure_is_known_consumed_and_fresh_request_only()
    test_oversized_or_overcount_output_fails_closed()
    test_exact_list_not_found_is_not_verified_empty()
    test_exact_list_identity_and_ambiguity_fail_closed()
    test_success_metadata_is_content_free_and_truthful()
    test_runtime_persists_only_content_free_summary()
    test_same_session_approval_query_once_boundary_and_retention()
    print("Apple reminders smoke passed")


if __name__ == "__main__":
    main()
