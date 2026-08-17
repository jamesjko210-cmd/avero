"""Approval-gated access to the shared macOS Reminders account.

The connector never creates, completes, or changes reminders. It returns at
most ``limit`` open items, using one extra returned record only to prove
truncation. Apple Event collection materialization is not independently
bounded or attested. An optional exact list name provides the data-minimised
path used by supervised V3 proofs.
The AppleScript source checks that Reminders is already running and contains no
launch request. The later Apple-event send is still a separate step, so this
does not attest that the check/send race has been eliminated.

Jarvis-owned reminders/timers are a separate feature whose Telegram API
acceptance is the automatable proof, while phone-display delivery remains
live-user confirmation. This connector only reads Apple's native reminders.

``_run_reminders_query`` is the mockable seam used by offline smoke tests.  A
real query is never made by the test suite.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
import re
import selectors
import signal
import subprocess
import time
import unicodedata
from typing import Any

from jarvis_v2.agent.failure_guidance import (
    PERSONAL_READ_RECOVERY_ACTION,
    declare_retryable_personal_read_failure,
)
from jarvis_v2.agent.types import ApprovalArgumentResolution, RiskLevel, ToolResult


DEFAULT_REMINDER_LIMIT = 5
MAX_REMINDERS = 10
MAX_LIST_NAME_CHARS = 80
MAX_REMINDER_TEXT_CHARS = 240
OSASCRIPT_PATH = "/usr/bin/osascript"
LOCAL_PATH_RE = re.compile(r"/(?:Users|private|var/folders|tmp)/[^\n\r;]*", re.IGNORECASE)

# Values are percent-escaped by AppleScript before using these separators.
_FIELD_SEP = ":::"
_RECORD_SEP = "|||"
_STATUS_RECORD = "__JARVIS_REMINDERS_STATUS_V1__"
_STATUS_ALL = "ALL"
_STATUS_FOUND = "FOUND"
_STATUS_NOT_FOUND = "NOT_FOUND"
_STATUS_AMBIGUOUS = "AMBIGUOUS"
_MAX_QUERY_OUTPUT_CHARS = (MAX_REMINDERS + 1) * (
    (MAX_LIST_NAME_CHARS * 3)
    + len(_FIELD_SEP)
    + (MAX_REMINDER_TEXT_CHARS * 3)
    + len(_RECORD_SEP)
) + 64
_MAX_STDOUT_BYTES = _MAX_QUERY_OUTPUT_CHARS * 4
_MAX_STDERR_BYTES = 4096
_PROCESS_TIMEOUT_SECONDS = 20.0
_PROCESS_TERMINATE_GRACE_SECONDS = 0.5

_REMINDERS_SCRIPT = f'''on replaceText(findText, replacementText, sourceText)
    set previousDelimiters to AppleScript's text item delimiters
    set AppleScript's text item delimiters to findText
    set sourceParts to text items of sourceText
    set AppleScript's text item delimiters to replacementText
    set encodedText to sourceParts as text
    set AppleScript's text item delimiters to previousDelimiters
    return encodedText
end replaceText

on encodeValue(sourceText)
    set encodedText to my replaceText("%", "%25", sourceText as text)
    set encodedText to my replaceText(":", "%3A", encodedText)
    set encodedText to my replaceText("|", "%7C", encodedText)
    return encodedText
end encodeValue

on boundedText(sourceText, maximumLength)
    set sourceText to sourceText as text
    if (length of sourceText) > maximumLength then
        return text 1 thru maximumLength of sourceText
    end if
    return sourceText
end boundedText

on run argv
    if (count of argv) is not 2 then error number -1703
    set requestedListName to item 1 of argv
    set requestedLimit to (item 2 of argv) as integer
    set fetchLimit to requestedLimit + 1
    if application "Reminders" is not running then error number -600
    tell application "Reminders"
        set theOutput to "{_STATUS_RECORD}{_FIELD_SEP}{_STATUS_ALL}{_RECORD_SEP}"
        set fetchedCount to 0
        if requestedListName is not "" then
            set matchedExactListCount to 0
            set matchedExactList to missing value
            considering case, diacriticals
                repeat with aList in lists
                    set fullListName to name of aList as text
                    if fullListName is requestedListName then
                        set matchedExactListCount to matchedExactListCount + 1
                        set matchedExactList to contents of aList
                    end if
                end repeat
            end considering
            if matchedExactListCount is 0 then
                return "{_STATUS_RECORD}{_FIELD_SEP}{_STATUS_NOT_FOUND}{_RECORD_SEP}"
            end if
            if matchedExactListCount is greater than 1 then
                return "{_STATUS_RECORD}{_FIELD_SEP}{_STATUS_AMBIGUOUS}{_RECORD_SEP}"
            end if
            set fullListName to name of matchedExactList as text
            set theOutput to "{_STATUS_RECORD}{_FIELD_SEP}{_STATUS_FOUND}{_RECORD_SEP}"
            repeat with aReminder in (reminders of matchedExactList whose completed is false)
                set listName to my boundedText(fullListName, {MAX_LIST_NAME_CHARS})
                set reminderName to my boundedText(name of aReminder as text, {MAX_REMINDER_TEXT_CHARS})
                set theOutput to theOutput & my encodeValue(listName) & "{_FIELD_SEP}" & my encodeValue(reminderName) & "{_RECORD_SEP}"
                set fetchedCount to fetchedCount + 1
                if fetchedCount is greater than or equal to fetchLimit then return theOutput
            end repeat
            return theOutput
        end if
        repeat with aList in lists
            set fullListName to name of aList as text
            repeat with aReminder in (reminders of aList whose completed is false)
                set listName to my boundedText(fullListName, {MAX_LIST_NAME_CHARS})
                set reminderName to my boundedText(name of aReminder as text, {MAX_REMINDER_TEXT_CHARS})
                set theOutput to theOutput & my encodeValue(listName) & "{_FIELD_SEP}" & my encodeValue(reminderName) & "{_RECORD_SEP}"
                set fetchedCount to fetchedCount + 1
                if fetchedCount is greater than or equal to fetchLimit then return theOutput
            end repeat
        end repeat
        return theOutput
    end tell
end run'''

# Test seam: callable(exact_list_name_or_empty, limit)->str.
_run_reminders_query = None  # type: ignore[assignment]


class AppleRemindersQueryError(RuntimeError):
    """Bounded connector failure carrying only a stable stage."""

    def __init__(self, stage: str):
        self.stage = str(stage or "query_failed")[:80]
        super().__init__(self.stage)


def _validated_query_args(args: dict[str, Any]) -> tuple[str, int]:
    unknown = sorted(set(args) - {"list_name", "limit"})
    if unknown:
        raise ValueError("unsupported Apple Reminders argument")

    raw_list_name = args.get("list_name", "")
    if not isinstance(raw_list_name, str):
        raise ValueError("Apple Reminders list name must be text")
    list_name = raw_list_name.strip()
    if len(list_name) > MAX_LIST_NAME_CHARS:
        raise ValueError("Apple Reminders list name is too long")
    if any(unicodedata.category(char) in {"Cc", "Cf"} for char in list_name):
        raise ValueError("Apple Reminders list name contains unsupported control characters")

    raw_limit = args.get("limit", DEFAULT_REMINDER_LIMIT)
    if isinstance(raw_limit, bool) or not isinstance(raw_limit, int):
        raise ValueError("Apple Reminders limit must be an integer")
    if not 1 <= raw_limit <= MAX_REMINDERS:
        raise ValueError("Apple Reminders limit is outside the supported range")
    return list_name, raw_limit


def resolve_apple_reminders_approval(
    args: dict[str, Any],
) -> ApprovalArgumentResolution | ToolResult:
    """Normalize and semantically validate the exact read before approval."""
    try:
        list_name, limit = _validated_query_args(args)
    except ValueError:
        return ToolResult(
            "apple_reminders",
            False,
            (
                f"Use an exact Apple Reminders list name and an integer limit from 1 to {MAX_REMINDERS}. "
                "No approval was queued and no Reminders data was read."
            ),
            {
                "failure_kind": "apple_reminders_approval_arguments_invalid",
                "failure_stage": "preapproval_argument_validation",
                "requires_confirmation": False,
                "executed_handler": False,
                "handler_invoked": False,
                "authorizes_execution": False,
                "approval_granted": False,
                "reads_personal_data": False,
                "reads_private_data": False,
                "reads_apple_reminders": False,
                "query_verified": False,
            },
        )
    return ApprovalArgumentResolution(
        {"list_name": list_name, "limit": limit},
        {"apple_reminders_arguments_bound_before_approval": True},
    )


@dataclass(frozen=True)
class _BoundedProcessResult:
    returncode: int | None
    stdout: bytes
    stderr: bytes
    timed_out: bool = False
    overflowed: bool = False
    process_reaped: bool = True
    post_start_failed: bool = False


def _terminate_and_reap(process: subprocess.Popen[bytes]) -> bool:
    """Terminate a started process group and always attempt to reap it."""
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except (OSError, ProcessLookupError):
            try:
                process.terminate()
            except OSError:
                pass
        try:
            process.wait(timeout=_PROCESS_TERMINATE_GRACE_SECONDS)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except (OSError, ProcessLookupError):
                try:
                    process.kill()
                except OSError:
                    pass
    try:
        process.wait(timeout=_PROCESS_TERMINATE_GRACE_SECONDS)
        return True
    except (OSError, subprocess.TimeoutExpired):
        return False


def _run_bounded_process(
    command: list[str],
    *,
    timeout: float = _PROCESS_TIMEOUT_SECONDS,
    stdout_cap: int = _MAX_STDOUT_BYTES,
    stderr_cap: int = _MAX_STDERR_BYTES,
) -> _BoundedProcessResult:
    """Stream capped output and stop/reap immediately on overflow or timeout."""
    process = subprocess.Popen(
        command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=False,
        text=False,
        bufsize=0,
        start_new_session=True,
    )
    selector = selectors.DefaultSelector()
    buffers = {"stdout": bytearray(), "stderr": bytearray()}
    caps = {"stdout": max(0, int(stdout_cap)), "stderr": max(0, int(stderr_cap))}
    overflowed = False
    timed_out = False
    post_start_failed = False
    deadline = time.monotonic() + max(0.01, float(timeout))
    try:
        assert process.stdout is not None and process.stderr is not None
        selector.register(process.stdout, selectors.EVENT_READ, "stdout")
        selector.register(process.stderr, selectors.EVENT_READ, "stderr")
        while selector.get_map():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                timed_out = True
                break
            events = selector.select(min(remaining, 0.1))
            if not events:
                if process.poll() is not None:
                    # Drain any bytes already buffered in the pipes.
                    events = [(key, selectors.EVENT_READ) for key in selector.get_map().values()]
                else:
                    continue
            for key, _ in events:
                try:
                    chunk = os.read(key.fileobj.fileno(), 4096)
                except OSError:
                    post_start_failed = True
                    break
                if not chunk:
                    selector.unregister(key.fileobj)
                    continue
                name = str(key.data)
                room = caps[name] - len(buffers[name])
                if room > 0:
                    buffers[name].extend(chunk[:room])
                if len(chunk) > max(0, room):
                    overflowed = True
                    break
            if overflowed or post_start_failed:
                break
        if not (overflowed or timed_out or post_start_failed):
            try:
                process.wait(timeout=max(0.01, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                timed_out = True
        reaped = _terminate_and_reap(process) if (overflowed or timed_out or post_start_failed) else True
        return _BoundedProcessResult(
            returncode=process.returncode,
            stdout=bytes(buffers["stdout"]),
            stderr=bytes(buffers["stderr"]),
            timed_out=timed_out,
            overflowed=overflowed,
            process_reaped=reaped,
            post_start_failed=post_start_failed,
        )
    except Exception:
        _terminate_and_reap(process)
        raise
    finally:
        selector.close()
        for stream in (process.stdout, process.stderr):
            if stream is not None:
                stream.close()


def _osascript_reminders(list_name: str, limit: int) -> _BoundedProcessResult:
    return _run_bounded_process(
        [OSASCRIPT_PATH, "-e", _REMINDERS_SCRIPT, "--", list_name, str(limit)]
    )


def _classify_process_failure(result: _BoundedProcessResult) -> str:
    stderr = bytes(result.stderr or b"").decode("utf-8", errors="replace").casefold()
    if "-600" in stderr or "isn't running" in stderr or "not running" in stderr:
        return "app_not_running"
    if "-1743" in stderr or "not authorized" in stderr or "not permitted" in stderr:
        return "permission_denied"
    return "query_failed"


def _real_reminders_query(list_name: str, limit: int) -> str:
    """Run one bounded read-only lookup and return framed stdout."""
    try:
        result = _osascript_reminders(list_name, limit)
    except FileNotFoundError as exc:
        raise AppleRemindersQueryError("executable_missing") from None
    except (OSError, UnicodeError) as exc:
        raise AppleRemindersQueryError("process_failed") from None
    if result.timed_out:
        raise AppleRemindersQueryError("query_timeout")
    if result.overflowed:
        raise AppleRemindersQueryError("output_too_large")
    if result.post_start_failed or not result.process_reaped:
        raise AppleRemindersQueryError("process_failed")
    if result.returncode != 0:
        raise AppleRemindersQueryError(_classify_process_failure(result))
    try:
        stdout = bytes(result.stdout or b"").decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        raise AppleRemindersQueryError("malformed_output") from None
    if len(stdout) > _MAX_QUERY_OUTPUT_CHARS:
        raise AppleRemindersQueryError("output_too_large")
    return stdout


def _query(list_name: str, limit: int) -> str:
    runner = _run_reminders_query or _real_reminders_query
    return runner(list_name, limit)


def _decode_value(value: str) -> str:
    # Reverse in the opposite order of encoding. A literal "%7C" becomes
    # "%257C" on the wire and therefore remains literal after one pass.
    return value.replace("%7C", "|").replace("%3A", ":").replace("%25", "%")


def _parse_reminders(
    raw: str, *, limit: int, exact_list_name: str = ""
) -> tuple[list[tuple[str, str]], bool, bool]:
    """Parse framed output and return the visible items plus truncation truth."""
    exact_list = bool(exact_list_name)
    if len(raw or "") > _MAX_QUERY_OUTPUT_CHARS:
        raise AppleRemindersQueryError("output_too_large")
    records = (raw or "").split(_RECORD_SEP)
    status_record = records.pop(0) if records else ""
    status_name, status_sep, status = status_record.partition(_FIELD_SEP)
    allowed_statuses = {
        _STATUS_ALL,
        _STATUS_FOUND,
        _STATUS_NOT_FOUND,
        _STATUS_AMBIGUOUS,
    }
    if status_name != _STATUS_RECORD or not status_sep or status not in allowed_statuses:
        raise AppleRemindersQueryError("malformed_output")
    if exact_list and status == _STATUS_ALL:
        raise AppleRemindersQueryError("malformed_output")
    if not exact_list and status != _STATUS_ALL:
        raise AppleRemindersQueryError("malformed_output")
    if status == _STATUS_AMBIGUOUS:
        if any(record.strip() for record in records):
            raise AppleRemindersQueryError("malformed_output")
        raise AppleRemindersQueryError("list_ambiguous")

    items: list[tuple[str, str]] = []
    for record in records:
        if not record or not record.strip():
            continue
        list_name, sep, reminder = record.partition(_FIELD_SEP)
        if not sep:
            raise AppleRemindersQueryError("malformed_output")
        list_name = _decode_value(list_name)
        reminder = _decode_value(reminder)
        if not reminder:
            raise AppleRemindersQueryError("malformed_output")
        if len(list_name) > MAX_LIST_NAME_CHARS or len(reminder) > MAX_REMINDER_TEXT_CHARS:
            raise AppleRemindersQueryError("malformed_output")
        if exact_list and list_name != exact_list_name:
            raise AppleRemindersQueryError("exact_list_mismatch")
        items.append((list_name, reminder))
        if len(items) > limit + 1:
            raise AppleRemindersQueryError("output_too_large")
    list_found = status != _STATUS_NOT_FOUND
    if not list_found and items:
        raise AppleRemindersQueryError("malformed_output")
    return items[:limit], len(items) > limit, list_found


def list_apple_reminders(
    *, list_name: str = "", limit: int = DEFAULT_REMINDER_LIMIT
) -> tuple[list[tuple[str, str]], bool]:
    """Return bounded open reminders and verified truncation state."""
    raw = _query(list_name, limit)
    items, truncated, list_found = _parse_reminders(
        raw, limit=limit, exact_list_name=list_name
    )
    if list_name and not list_found:
        raise AppleRemindersQueryError("list_not_found")
    return items, truncated


def _safe_metadata(**extra: Any) -> dict[str, Any]:
    base = {
        "calls_model": False,
        "calls_external_service": False,
        "calls_external_services": False,
        "executes_tools": False,
        "reads_personal_data": True,
        "reads_private_data": True,
        "executes_side_effect": False,
        "external_side_effect": False,
        "writes_files": False,
        "writes_database": False,
        "writes_memory": False,
        "writes_notes": False,
        "queues_approval": False,
        "requires_approval": False,
        "controls_computer": False,
        "reads_apple_reminders": True,
        "source_scope": "shared_macos_account",
        "uses_v2_state": False,
        "v2_state_read": False,
        "returned_content_bounded": True,
        "apple_event_materialization_bounded": False,
        "result_content_in_metadata": False,
        "suppress_output_persistence": True,
        "app_launch_requested": False,
        "app_running_precheck_configured": True,
        "app_launch_race_eliminated": False,
    }
    base.update(extra)
    return base


def _safe_text(value: str, *, limit: int) -> str:
    chars: list[str] = []
    for char in str(value or ""):
        category = unicodedata.category(char)
        if char in "\r\n\t":
            chars.append(" ")
        elif category not in {"Cc", "Cf"}:
            chars.append(char)
    text = " ".join("".join(chars).strip().split())[:limit]
    return LOCAL_PATH_RE.sub("<local-path>", text)


def _apple_reminders_recovery_message(stage: str = "query_failed") -> str:
    if stage == "app_not_running":
        return (
            "I couldn't read Apple Reminders because the Reminders app was not already running. "
            "The approved attempt is consumed and cannot be replayed. Open Reminders yourself, then "
            "submit the same bounded read as a fresh request and review its new one-shot approval. "
            "Jarvis did not request an app launch, but this check cannot attest a race-free no-launch "
            "boundary."
        )
    if stage == "permission_denied":
        return (
            "macOS denied Apple Reminders access. In System Settings > Privacy & Security > Automation, "
            "review Reminders access for the Terminal process you used, then retry only with the operator present."
        )
    if stage == "executable_missing":
        return "Apple Reminders reads require macOS and the system /usr/bin/osascript executable."
    if stage in {"malformed_output", "output_too_large"}:
        return (
            "Apple Reminders returned an invalid or oversized bounded result, so Jarvis discarded it. "
            "Stop this proof and do not treat partial output as current data."
        )
    if stage == "query_timeout":
        return (
            "The bounded Apple Reminders read timed out with no verified result. "
            "Do not retry automatically; check the app and permission state first."
        )
    if stage == "list_not_found":
        return (
            "That exact Apple Reminders list was not found, so Jarvis did not substitute another list. "
            "Check the exact list name, then submit a new bounded read."
        )
    if stage == "list_ambiguous":
        return (
            "More than one Apple Reminders list has that exact case-sensitive name, so Jarvis "
            "stopped without choosing one. Rename or remove the duplicate proof list, then submit "
            "a new bounded read."
        )
    if stage == "exact_list_mismatch":
        return (
            "Apple Reminders returned data from a list that did not exactly match the approved "
            "case-sensitive list name, so Jarvis discarded the result. Stop this proof; do not "
            "treat the mismatched result as current data."
        )
    return (
        "I couldn't verify your Apple Reminders right now. Open Reminders yourself and keep it open, "
        "then check Reminders access in macOS System Settings > Privacy & Security > Automation. "
        f"{PERSONAL_READ_RECOVERY_ACTION}"
    )


def _apple_reminders_recovery_action(stage: str) -> tuple[str, tuple[str, ...]]:
    if stage == "app_not_running":
        return (
            "Open Reminders yourself, then submit the same bounded read as a fresh request and review its new one-shot approval.",
            (),
        )
    if stage == "permission_denied":
        return (
            "In System Settings > Privacy & Security > Automation, review Reminders access for the Terminal process you used, then retry only with the operator present.",
            (),
        )
    if stage == "executable_missing":
        return "Apple Reminders reads require macOS and the system /usr/bin/osascript executable.", ()
    if stage in {"malformed_output", "output_too_large"}:
        return "Stop this proof and do not treat partial output as current data.", ()
    if stage == "query_timeout":
        return "Do not retry automatically; check the app and permission state first.", ()
    if stage == "list_not_found":
        return "Check the exact list name, then submit a new bounded read.", ()
    if stage == "list_ambiguous":
        return "Rename or remove the duplicate proof list, then submit a new bounded read.", ()
    if stage == "exact_list_mismatch":
        return "Stop this proof; do not treat the mismatched result as current data.", ()
    return PERSONAL_READ_RECOVERY_ACTION, ("setup check",)


def make_apple_reminders_tools(config):
    def boundary_receipt(*, exact_list: bool, returned: int, limit: int, truncated: bool) -> str:
        return (
            "Boundary: returned content bounded: yes; "
            f"exact list identity verified: {'yes' if exact_list else 'not requested'}; "
            f"returned {returned} of maximum {limit}; additional open items: "
            f"{'yes' if truncated else 'no'}; Apple Event materialization bounded: no (not attested)."
        )

    def apple_reminders(args: dict[str, Any]) -> ToolResult:
        try:
            list_name, limit = _validated_query_args(args)
        except ValueError:
            output = (
                f"Use an exact Apple Reminders list name and an integer limit from 1 to {MAX_REMINDERS}. "
                "No Reminders data was read."
            )
            return ToolResult(
                "apple_reminders",
                False,
                output,
                _safe_metadata(
                    reads_personal_data=False,
                    reads_private_data=False,
                    reads_apple_reminders=False,
                    returned_content_bounded=False,
                    failure_kind="apple_reminders_invalid_arguments",
                    failure_stage="argument_validation",
                    query_verified=False,
                    verified_empty=False,
                    handler_invoked=False,
                    app_running_precheck_status="unknown",
                    app_running_precheck_performed=False,
                ),
            )

        exact_list = bool(list_name)
        try:
            items, truncated = list_apple_reminders(list_name=list_name, limit=limit)
        except Exception as exc:
            stage = exc.stage if isinstance(exc, AppleRemindersQueryError) else "query_failed"
            failure_output = _apple_reminders_recovery_message(stage)
            recovery_action, recovery_commands = _apple_reminders_recovery_action(stage)
            return ToolResult(
                "apple_reminders",
                False,
                failure_output,
                declare_retryable_personal_read_failure(
                    _safe_metadata(
                        failure_kind=f"apple_reminders_{stage}",
                        lookup_status="unavailable",
                        reminder_read_status="unavailable",
                        query_verified=False,
                        verified_empty=False,
                        failure_stage=stage,
                        error_type=type(exc).__name__[:80],
                        retryable=stage
                        not in {
                            "malformed_output",
                            "output_too_large",
                            "list_ambiguous",
                            "exact_list_mismatch",
                        },
                        exact_list_filter_applied=exact_list,
                        exact_list_identity_verified=False,
                        requested_limit=limit,
                        returned_reminder_count=0,
                        fetch_limit=limit + 1,
                        truncated=False,
                        app_running_precheck_status=(
                            "not_running"
                            if stage == "app_not_running"
                            else "passed"
                            if stage
                            in {
                                "list_not_found",
                                "list_ambiguous",
                                "exact_list_mismatch",
                                "malformed_output",
                            }
                            else "unknown"
                        ),
                        **(
                            {
                                "app_running_precheck_performed": True,
                                "app_running_precheck": True,
                                "app_running_precheck_passed": stage != "app_not_running",
                            }
                            if stage
                            in {
                                "app_not_running",
                                "list_not_found",
                                "list_ambiguous",
                                "exact_list_mismatch",
                                "malformed_output",
                            }
                            else {}
                        ),
                    ),
                    output=failure_output,
                    action=recovery_action,
                    commands=recovery_commands,
                ),
            )

        if not items:
            status_line = (
                "There are no open reminders in that exact Apple Reminders list."
                if exact_list
                else "You have no open reminders in the Apple Reminders app."
            )
            output = status_line + "\n\n" + boundary_receipt(
                exact_list=exact_list,
                returned=0,
                limit=limit,
                truncated=False,
            )
            return ToolResult(
                "apple_reminders",
                True,
                output,
                _safe_metadata(
                    reminder_count=0,
                    list_count=0,
                    reminder_read_status="verified_empty",
                    query_verified=True,
                    verified_empty=True,
                    retryable=False,
                    exact_list_filter_applied=exact_list,
                    exact_list_identity_verified=exact_list,
                    requested_limit=limit,
                    returned_reminder_count=0,
                    fetch_limit=limit + 1,
                    truncated=False,
                    app_running_precheck_status="passed",
                    app_running_precheck_performed=True,
                    app_running_precheck=True,
                    app_running_precheck_passed=True,
                ),
            )

        by_list: dict[str, list[str]] = {}
        for raw_list_name, reminder in items:
            safe_list_name = _safe_text(raw_list_name, limit=MAX_LIST_NAME_CHARS) or "Reminders"
            by_list.setdefault(safe_list_name, []).append(
                _safe_text(reminder, limit=MAX_REMINDER_TEXT_CHARS)
            )
        lines = [f"📋 Apple Reminders ({len(items)} open):"]
        for current_list_name in sorted(by_list):
            lines.append(f"\n{current_list_name}:")
            for reminder in by_list[current_list_name]:
                lines.append(f"  • {reminder}")
        lines.extend(
            [
                "",
                boundary_receipt(
                    exact_list=exact_list,
                    returned=len(items),
                    limit=limit,
                    truncated=truncated,
                ),
            ]
        )
        return ToolResult(
            "apple_reminders",
            True,
            "\n".join(lines),
            _safe_metadata(
                reminder_count=len(items),
                list_count=len(by_list),
                reminder_read_status="verified_nonempty",
                query_verified=True,
                verified_empty=False,
                truncated=truncated,
                retryable=False,
                exact_list_filter_applied=exact_list,
                exact_list_identity_verified=exact_list,
                requested_limit=limit,
                returned_reminder_count=len(items),
                fetch_limit=limit + 1,
                app_running_precheck_status="passed",
                app_running_precheck_performed=True,
                app_running_precheck=True,
                app_running_precheck_passed=True,
            ),
        )

    from jarvis_v2.tools.registry import (
        Tool,
        ToolArgumentContract,
        ToolArgumentSpec,
        ToolArgumentType,
        TOOL_ARGUMENT_CONTRACT_VERSION,
    )

    argument_contract = ToolArgumentContract(
        version=TOOL_ARGUMENT_CONTRACT_VERSION,
        fields=(
            ToolArgumentSpec("list_name", frozenset({ToolArgumentType.STRING}), False),
            ToolArgumentSpec(
                "limit",
                frozenset({ToolArgumentType.INTEGER}),
                False,
                1,
                MAX_REMINDERS,
            ),
        ),
        allow_unknown=False,
    )
    return [
        Tool(
            "apple_reminders",
            "List a bounded number of open macOS Reminders, optionally from one exact list. Args: optional list_name and limit (1-10).",
            RiskLevel.PERSONAL_DATA,
            apple_reminders,
            "personal",
            argument_contract=argument_contract,
            approval_argument_resolver=resolve_apple_reminders_approval,
            approval_argument_contract=argument_contract,
        )
    ]
