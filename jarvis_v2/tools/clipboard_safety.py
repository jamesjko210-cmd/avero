"""Fail-closed helpers for temporary plain-text clipboard use.

``PlainTextClipboardTransaction`` is the safe API for new callers.  It refuses
rich/non-text clipboards, uses the pasteboard change count as an ownership
token, skips restoration if another writer has replaced Jarvis's staged text,
and verifies the exact plain-text value after restoration.

Clipboard and staged text only travel over subprocess stdin/stdout.  They are
never placed in argv or exception messages.  The two legacy snapshot/restore
functions remain temporarily for connector migration; unlike the transaction
API, they cannot provide an ownership guard across separate calls.
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable
from contextlib import contextmanager
from typing import Any, Iterator


PLAIN_CLIPBOARD_TYPES = frozenset(
    {
        "«class ut16»",
        "«class utf8»",
        "«class text»",
        "string",
        "text",
        "unicode text",
    }
)

_CHANGE_COUNT_JXA = """
ObjC.import("AppKit");
function run() {
    return String($.NSPasteboard.generalPasteboard.changeCount);
}
""".strip()

_CONDITIONAL_PLAIN_TEXT_WRITE_JXA = """
ObjC.import("AppKit");
ObjC.import("Foundation");
function run(argv) {
    if (argv.length !== 1 || !/^-?[0-9]+$/.test(String(argv[0]))) {
        return "INVALID_EXPECTED_COUNT";
    }
    var expected = Number(argv[0]);
    var pasteboard = $.NSPasteboard.generalPasteboard;
    if (Number(pasteboard.changeCount) !== expected) {
        return "STALE";
    }
    var input = $.NSFileHandle.fileHandleWithStandardInput.readDataToEndOfFile;
    var value = $.NSString.alloc.initWithDataEncoding(input, $.NSUTF8StringEncoding);
    if (value === null) {
        return "INVALID_UTF8";
    }
    pasteboard.clearContents;
    if (!pasteboard.setStringForType(value, $.NSPasteboardTypeString)) {
        return "WRITE_FAILED";
    }
    return "WRITTEN:" + String(pasteboard.changeCount);
}
""".strip()

_CHANGE_COUNT_COMMAND = [
    "osascript",
    "-l",
    "JavaScript",
    "-e",
    _CHANGE_COUNT_JXA,
]


class ClipboardSafetyError(RuntimeError):
    def __init__(self, stage: str) -> None:
        self.stage = stage
        super().__init__(stage)


def privacy_boundary_metadata(
    *,
    snapshot_attempted: bool,
    snapshot_captured: bool,
    replacement_attempted: bool,
    text_restored: bool | None,
) -> dict[str, Any]:
    """Return content-free facts about temporary clipboard custody.

    ``clipboard_fully_restored`` is intentionally false after replacement:
    these helpers preserve the verified plain-text value, not the original
    pasteboard representation set.  A failed restore means private staged text
    may remain in the clipboard and must be treated as an unknown side effect.
    """

    clipboard_outcome_known = not replacement_attempted or text_restored is True
    if not replacement_attempted:
        restoration_scope = "untouched"
        fully_restored = True
    elif text_restored is True:
        restoration_scope = "plain_text_value_only"
        fully_restored = False
    elif text_restored is False:
        restoration_scope = "unknown"
        fully_restored = False
    else:
        restoration_scope = "pending"
        fully_restored = False
    facts = {
        "clipboard_snapshot_attempted": snapshot_attempted,
        "clipboard_snapshot_captured": snapshot_captured,
        "clipboard_replacement_attempted": replacement_attempted,
        "clipboard_text_restored": text_restored,
        "clipboard_fully_restored": fully_restored,
        "clipboard_restoration_scope": restoration_scope,
        "clipboard_outcome_known": clipboard_outcome_known,
        "clipboard_private_content_possible": (
            replacement_attempted and text_restored is not True
        ),
    }
    return {**facts, "clipboard_privacy_boundary": dict(facts)}


def _plain_text_type_report(raw: Any) -> bool:
    text = str(raw or "").strip()
    if text.startswith("{") and text.endswith("}"):
        text = text[1:-1].strip()
    parts = [part.strip().casefold() for part in text.split(",") if part.strip()]
    if not parts or len(parts) % 2:
        raise ClipboardSafetyError("clipboard_inspection")
    for index in range(0, len(parts), 2):
        if (
            parts[index] not in PLAIN_CLIPBOARD_TYPES
            or not parts[index + 1].isdigit()
        ):
            return False
    return True


def _run_change_count(run: Callable[..., Any]) -> int:
    try:
        result = run(
            list(_CHANGE_COUNT_COMMAND),
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception as exc:
        raise ClipboardSafetyError("clipboard_change_count") from exc
    if getattr(result, "returncode", 1) != 0:
        raise ClipboardSafetyError("clipboard_change_count")
    raw = str(getattr(result, "stdout", "") or "").strip()
    if not raw or raw.lstrip("-").isdigit() is False:
        raise ClipboardSafetyError("clipboard_change_count")
    try:
        return int(raw)
    except (TypeError, ValueError) as exc:
        raise ClipboardSafetyError("clipboard_change_count") from exc


def _inspect_plain_text_at_count(
    expected_change_count: int,
    run: Callable[..., Any],
) -> None:
    if _run_change_count(run) != expected_change_count:
        raise ClipboardSafetyError("clipboard_changed")
    try:
        inspected = run(
            ["osascript", "-e", "clipboard info"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception as exc:
        raise ClipboardSafetyError("clipboard_inspection") from exc
    if getattr(inspected, "returncode", 1) != 0:
        raise ClipboardSafetyError("clipboard_inspection")
    if _run_change_count(run) != expected_change_count:
        raise ClipboardSafetyError("clipboard_changed")
    if not _plain_text_type_report(getattr(inspected, "stdout", "")):
        raise ClipboardSafetyError("clipboard_rich_content")


def _read_plain_text_at_count(
    expected_change_count: int,
    run: Callable[..., Any],
) -> str:
    if _run_change_count(run) != expected_change_count:
        raise ClipboardSafetyError("clipboard_changed")
    try:
        captured = run(
            ["pbpaste"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception as exc:
        raise ClipboardSafetyError("clipboard_read") from exc
    if getattr(captured, "returncode", 1) != 0:
        raise ClipboardSafetyError("clipboard_read")
    if _run_change_count(run) != expected_change_count:
        raise ClipboardSafetyError("clipboard_changed")
    return str(getattr(captured, "stdout", "") or "")


def _conditional_plain_text_write(
    value: str,
    *,
    expected_change_count: int,
    run: Callable[..., Any],
    failure_stage: str,
) -> int:
    command = [
        "osascript",
        "-l",
        "JavaScript",
        "-e",
        _CONDITIONAL_PLAIN_TEXT_WRITE_JXA,
        str(expected_change_count),
    ]
    try:
        result = run(
            command,
            input=value,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception as exc:
        raise ClipboardSafetyError(failure_stage) from exc
    if getattr(result, "returncode", 1) != 0:
        raise ClipboardSafetyError(failure_stage)
    status = str(getattr(result, "stdout", "") or "").strip()
    if status == "STALE":
        raise ClipboardSafetyError("clipboard_ownership_lost")
    prefix = "WRITTEN:"
    if not status.startswith(prefix):
        raise ClipboardSafetyError(failure_stage)
    raw_count = status.removeprefix(prefix)
    if not raw_count or raw_count.lstrip("-").isdigit() is False:
        raise ClipboardSafetyError(failure_stage)
    changed_count = int(raw_count)
    if changed_count == expected_change_count:
        raise ClipboardSafetyError(failure_stage)
    return changed_count


class PlainTextClipboardTransaction:
    """Own one temporary plain-text clipboard replacement.

    Use as a context manager.  A transaction snapshots only a stable, plain
    clipboard.  ``stage`` conditionally replaces that snapshot and records the
    resulting change count.  On exit, restoration happens only while that
    change count and the exact staged value still match.

    The class deliberately has no public clipboard-content attributes and its
    repr never includes either private value.
    """

    __slots__ = (
        "_run",
        "_saved_text",
        "_staged_text",
        "_snapshot_change_count",
        "_owned_change_count",
        "_write_may_have_started",
        "_snapshot_attempted",
        "_snapshot_captured",
        "_entered",
        "_closed",
        "_restored",
    )

    def __init__(self, run: Callable[..., Any] = subprocess.run) -> None:
        self._run = run
        self._saved_text: str | None = None
        self._staged_text: str | None = None
        self._snapshot_change_count: int | None = None
        self._owned_change_count: int | None = None
        self._write_may_have_started = False
        self._snapshot_attempted = False
        self._snapshot_captured = False
        self._entered = False
        self._closed = False
        self._restored = False

    def __repr__(self) -> str:
        return (
            "PlainTextClipboardTransaction("
            f"entered={self._entered}, staged={self._owned_change_count is not None}, "
            f"closed={self._closed}, restored={self._restored})"
        )

    @property
    def staged(self) -> bool:
        return self._owned_change_count is not None and not self._closed

    @property
    def restored(self) -> bool:
        return self._restored

    def privacy_metadata(self) -> dict[str, Any]:
        """Return content-free custody facts for connector metadata/handoffs."""

        text_restored: bool | None
        if self._restored:
            text_restored = True
        elif self._closed and self._write_may_have_started:
            text_restored = False
        else:
            text_restored = None
        return privacy_boundary_metadata(
            snapshot_attempted=self._snapshot_attempted,
            snapshot_captured=self._snapshot_captured,
            replacement_attempted=self._write_may_have_started,
            text_restored=text_restored,
        )

    def __enter__(self) -> PlainTextClipboardTransaction:
        if self._entered or self._closed:
            raise ClipboardSafetyError("clipboard_transaction_state")
        self._snapshot_attempted = True
        snapshot_count = _run_change_count(self._run)
        _inspect_plain_text_at_count(snapshot_count, self._run)
        saved_text = _read_plain_text_at_count(snapshot_count, self._run)
        if _run_change_count(self._run) != snapshot_count:
            raise ClipboardSafetyError("clipboard_changed")
        self._snapshot_change_count = snapshot_count
        self._saved_text = saved_text
        self._snapshot_captured = True
        self._entered = True
        return self

    def stage(self, value: str) -> None:
        if (
            not self._entered
            or self._closed
            or self._snapshot_change_count is None
            or self._saved_text is None
            or self._owned_change_count is not None
        ):
            raise ClipboardSafetyError("clipboard_transaction_state")
        if type(value) is not str:
            raise ClipboardSafetyError("clipboard_invalid_text")

        # The helper checks the ownership token again immediately before its
        # clear/write pair.  Content is supplied only over stdin.
        self._staged_text = value
        self._write_may_have_started = True
        try:
            owned_count = _conditional_plain_text_write(
                value,
                expected_change_count=self._snapshot_change_count,
                run=self._run,
                failure_stage="clipboard_write",
            )
        except ClipboardSafetyError as exc:
            # STALE is returned before the helper's clear/write pair.  Do not
            # attempt to "adopt" or restore a newer clipboard even if its text
            # happens to equal the value Jarvis intended to stage.
            if exc.stage == "clipboard_ownership_lost":
                self._write_may_have_started = False
            raise
        self._owned_change_count = owned_count
        _inspect_plain_text_at_count(owned_count, self._run)
        staged_value = _read_plain_text_at_count(owned_count, self._run)
        if staged_value != value:
            raise ClipboardSafetyError("clipboard_stage_verification")

    def restage(self, value: str) -> None:
        """Replace Jarvis's currently owned staged text without losing custody.

        The original snapshot remains the eventual restore target. A restage is
        allowed only while both the pasteboard change count and exact staged
        value still match this transaction. Any drift closes the transaction and
        deliberately skips restoration so a newer user/application clipboard is
        never overwritten.
        """
        if (
            not self._entered
            or self._closed
            or self._snapshot_change_count is None
            or self._saved_text is None
            or self._owned_change_count is None
            or self._staged_text is None
        ):
            raise ClipboardSafetyError("clipboard_transaction_state")
        if type(value) is not str:
            raise ClipboardSafetyError("clipboard_invalid_text")

        owned_count = self._owned_change_count
        try:
            if _run_change_count(self._run) != owned_count:
                raise ClipboardSafetyError("clipboard_ownership_lost")
            current_text = _read_plain_text_at_count(owned_count, self._run)
            if current_text != self._staged_text:
                raise ClipboardSafetyError("clipboard_ownership_lost")
        except ClipboardSafetyError:
            self._closed = True
            raise

        self._staged_text = value
        self._write_may_have_started = True
        try:
            new_owned_count = _conditional_plain_text_write(
                value,
                expected_change_count=owned_count,
                run=self._run,
                failure_stage="clipboard_restage",
            )
        except ClipboardSafetyError:
            # A failed helper may have stopped before or after its write. Never
            # guess ownership and never restore across that uncertainty.
            self._owned_change_count = None
            self._closed = True
            raise

        self._owned_change_count = new_owned_count
        try:
            _inspect_plain_text_at_count(new_owned_count, self._run)
            restaged_value = _read_plain_text_at_count(new_owned_count, self._run)
        except ClipboardSafetyError as exc:
            self._closed = True
            raise ClipboardSafetyError("clipboard_restage_verification") from exc
        if restaged_value != value:
            self._closed = True
            raise ClipboardSafetyError("clipboard_restage_verification")

    def _adopt_or_clear_uncertain_stage(self) -> None:
        if (
            not self._write_may_have_started
            or self._owned_change_count is not None
            or self._snapshot_change_count is None
        ):
            return
        current_count = _run_change_count(self._run)
        if current_count == self._snapshot_change_count:
            self._write_may_have_started = False
            return
        if self._staged_text is None:
            raise ClipboardSafetyError("clipboard_ownership_lost")
        try:
            current_text = _read_plain_text_at_count(current_count, self._run)
        except ClipboardSafetyError as exc:
            raise ClipboardSafetyError("clipboard_ownership_lost") from exc
        if current_text != self._staged_text:
            raise ClipboardSafetyError("clipboard_ownership_lost")
        self._owned_change_count = current_count

    def restore(self) -> None:
        if not self._entered or self._closed:
            raise ClipboardSafetyError("clipboard_transaction_state")
        if self._saved_text is None or self._snapshot_change_count is None:
            raise ClipboardSafetyError("clipboard_transaction_state")

        try:
            self._adopt_or_clear_uncertain_stage()
        except ClipboardSafetyError:
            self._closed = True
            raise
        if self._owned_change_count is None:
            self._closed = True
            return
        if _run_change_count(self._run) != self._owned_change_count:
            self._closed = True
            raise ClipboardSafetyError("clipboard_ownership_lost")
        if self._staged_text is None:
            self._closed = True
            raise ClipboardSafetyError("clipboard_ownership_lost")
        try:
            current_text = _read_plain_text_at_count(
                self._owned_change_count,
                self._run,
            )
        except ClipboardSafetyError as exc:
            self._closed = True
            raise ClipboardSafetyError("clipboard_ownership_lost") from exc
        if current_text != self._staged_text:
            self._closed = True
            raise ClipboardSafetyError("clipboard_ownership_lost")

        try:
            restored_count = _conditional_plain_text_write(
                self._saved_text,
                expected_change_count=self._owned_change_count,
                run=self._run,
                failure_stage="clipboard_restore",
            )
        except ClipboardSafetyError:
            self._closed = True
            raise
        try:
            _inspect_plain_text_at_count(restored_count, self._run)
            restored_text = _read_plain_text_at_count(restored_count, self._run)
        except ClipboardSafetyError as exc:
            self._closed = True
            raise ClipboardSafetyError("clipboard_restore_verification") from exc
        self._closed = True
        if restored_text != self._saved_text:
            raise ClipboardSafetyError("clipboard_restore_verification")
        self._restored = True

    def __exit__(self, exc_type, exc, traceback) -> bool:
        del exc_type, traceback
        if not self._closed:
            try:
                self.restore()
            except ClipboardSafetyError as restore_error:
                if exc is not None:
                    raise restore_error from exc
                raise
        return False


@contextmanager
def temporary_plain_text_clipboard(
    value: str,
    *,
    run: Callable[..., Any] = subprocess.run,
) -> Iterator[PlainTextClipboardTransaction]:
    """Yield a verified temporary plain-text clipboard replacement."""

    with PlainTextClipboardTransaction(run=run) as transaction:
        transaction.stage(value)
        yield transaction


def snapshot_plain_text_clipboard(
    run: Callable[..., Any] = subprocess.run,
) -> str:
    try:
        inspected = run(
            ["osascript", "-e", "clipboard info"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception as exc:
        raise ClipboardSafetyError("clipboard_inspection") from exc
    if getattr(inspected, "returncode", 1) != 0:
        raise ClipboardSafetyError("clipboard_inspection")
    if not _plain_text_type_report(getattr(inspected, "stdout", "")):
        raise ClipboardSafetyError("clipboard_rich_content")
    try:
        captured = run(
            ["pbpaste"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception as exc:
        raise ClipboardSafetyError("clipboard_read") from exc
    if getattr(captured, "returncode", 1) != 0:
        raise ClipboardSafetyError("clipboard_read")
    return str(getattr(captured, "stdout", "") or "")


def restore_plain_text_clipboard(
    saved: str,
    run: Callable[..., Any] = subprocess.run,
) -> None:
    try:
        restored = run(["pbcopy"], input=saved, text=True, timeout=5)
    except Exception as exc:
        raise ClipboardSafetyError("clipboard_restore") from exc
    if getattr(restored, "returncode", 1) != 0:
        raise ClipboardSafetyError("clipboard_restore")
