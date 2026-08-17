"""Standalone mocked smoke tests for the shared clipboard transaction."""

from __future__ import annotations

from types import SimpleNamespace

from jarvis_v2.tools import clipboard_safety as cs


PLAIN_INFO = "«class ut16», 8, «class utf8», 4, string, 4"
RICH_INFO = "«class RTF », 200, «class utf8», 20, string, 20"


class FakePasteboardRun:
    def __init__(self, text: str, *, info: str = PLAIN_INFO) -> None:
        self.text = text
        self.info = info
        self.change_count = 10
        self.calls: list[tuple[list[str], dict]] = []
        self.corrupt_next_read = False
        self.corrupt_after_write_number: int | None = None
        self.fail_next_write = False
        self.stale_next_write = False
        self.drift_after_info = False
        self.write_count = 0

    def external_copy(self, value: str, *, info: str = PLAIN_INFO) -> None:
        self.text = value
        self.info = info
        self.change_count += 1

    def __call__(self, command, **kwargs):
        argv = list(command)
        self.calls.append((argv, dict(kwargs)))
        if argv == cs._CHANGE_COUNT_COMMAND:
            return SimpleNamespace(
                returncode=0,
                stdout=str(self.change_count),
                stderr="",
            )
        if argv == ["osascript", "-e", "clipboard info"]:
            result = SimpleNamespace(returncode=0, stdout=self.info, stderr="")
            if self.drift_after_info:
                self.drift_after_info = False
                self.external_copy("newer user clipboard")
            return result
        if argv == ["pbpaste"]:
            value = self.text
            if self.corrupt_next_read:
                self.corrupt_next_read = False
                value = value + "-corrupt"
            return SimpleNamespace(returncode=0, stdout=value, stderr="")
        if (
            argv[:5]
            == ["osascript", "-l", "JavaScript", "-e", cs._CONDITIONAL_PLAIN_TEXT_WRITE_JXA]
            and len(argv) == 6
        ):
            expected = int(argv[5])
            if self.stale_next_write or expected != self.change_count:
                self.stale_next_write = False
                return SimpleNamespace(returncode=0, stdout="STALE", stderr="")
            if self.fail_next_write:
                self.fail_next_write = False
                return SimpleNamespace(
                    returncode=1,
                    stdout="",
                    stderr="PRIVATE /\x55sers/example/clipboard-secret.log",
                )
            self.text = kwargs.get("input")
            self.info = PLAIN_INFO
            self.change_count += 1
            self.write_count += 1
            if self.corrupt_after_write_number == self.write_count:
                self.corrupt_next_read = True
            return SimpleNamespace(
                returncode=0,
                stdout=f"WRITTEN:{self.change_count}",
                stderr="",
            )
        raise AssertionError(f"unexpected subprocess call: {argv!r}")


def _assert_no_content_in_argv(
    fake: FakePasteboardRun,
    *private_values: str,
) -> None:
    argv_text = "\n".join(repr(command) for command, _kwargs in fake.calls)
    for value in private_values:
        if value and value in argv_text:
            raise SystemExit(f"clipboard content leaked into argv: {argv_text}")


def test_happy_path_restores_exact_text_without_argv_leak() -> None:
    original = "user clipboard\n한글\u0000tail"
    staged = "PRIVATE /\x55sers/example/message.txt\nsend this"
    fake = FakePasteboardRun(original)
    with cs.temporary_plain_text_clipboard(staged, run=fake) as transaction:
        if fake.text != staged or not transaction.staged:
            raise SystemExit(f"clipboard staging did not verify: {transaction!r}")
        if original in repr(transaction) or staged in repr(transaction):
            raise SystemExit(f"transaction repr leaked clipboard content: {transaction!r}")
    if fake.text != original or not transaction.restored:
        raise SystemExit(
            f"clipboard transaction did not restore exact text: {fake.text!r}"
        )
    _assert_no_content_in_argv(fake, original, staged)
    privacy = transaction.privacy_metadata()
    if (
        privacy.get("clipboard_snapshot_attempted") is not True
        or privacy.get("clipboard_snapshot_captured") is not True
        or privacy.get("clipboard_replacement_attempted") is not True
        or privacy.get("clipboard_text_restored") is not True
        or privacy.get("clipboard_restoration_scope") != "plain_text_value_only"
        or privacy.get("clipboard_fully_restored") is not False
        or original in str(privacy)
        or staged in str(privacy)
    ):
        raise SystemExit(f"clipboard privacy metadata drifted: {privacy}")
    conditional_calls = [
        (argv, kwargs)
        for argv, kwargs in fake.calls
        if cs._CONDITIONAL_PLAIN_TEXT_WRITE_JXA in argv
    ]
    if len(conditional_calls) != 2:
        raise SystemExit(f"expected one stage and one restore: {conditional_calls}")
    if [kwargs.get("input") for _argv, kwargs in conditional_calls] != [
        staged,
        original,
    ]:
        raise SystemExit("private values must travel only through subprocess stdin")


def test_rich_clipboard_refuses_before_write() -> None:
    fake = FakePasteboardRun("rich text projection", info=RICH_INFO)
    try:
        with cs.temporary_plain_text_clipboard("private message", run=fake):
            raise AssertionError("rich clipboard entered transaction body")
    except cs.ClipboardSafetyError as exc:
        if exc.stage != "clipboard_rich_content":
            raise SystemExit(f"rich clipboard stage drifted: {exc.stage}")
    else:
        raise SystemExit("rich clipboard was accepted")
    if any(cs._CONDITIONAL_PLAIN_TEXT_WRITE_JXA in argv for argv, _ in fake.calls):
        raise SystemExit("rich clipboard refusal performed a write")


def test_snapshot_change_fails_before_write() -> None:
    fake = FakePasteboardRun("old user clipboard")
    fake.drift_after_info = True
    try:
        with cs.temporary_plain_text_clipboard("private message", run=fake):
            raise AssertionError("drifting clipboard entered transaction body")
    except cs.ClipboardSafetyError as exc:
        if exc.stage != "clipboard_changed":
            raise SystemExit(f"snapshot drift stage wrong: {exc.stage}")
    else:
        raise SystemExit("snapshot drift was accepted")
    if fake.text != "newer user clipboard":
        raise SystemExit("snapshot drift overwrote the newer clipboard")
    if any(cs._CONDITIONAL_PLAIN_TEXT_WRITE_JXA in argv for argv, _ in fake.calls):
        raise SystemExit("snapshot drift performed a write")


def test_ownership_loss_skips_restore_and_preserves_newer_clipboard() -> None:
    original = "old user clipboard"
    staged = "private staged message"
    newer = "newer user clipboard"
    fake = FakePasteboardRun(original)
    try:
        with cs.temporary_plain_text_clipboard(staged, run=fake):
            fake.external_copy(newer)
    except cs.ClipboardSafetyError as exc:
        if exc.stage != "clipboard_ownership_lost":
            raise SystemExit(f"ownership-loss stage drifted: {exc.stage}")
        if original in str(exc) or staged in str(exc) or newer in str(exc):
            raise SystemExit(f"ownership-loss error leaked private content: {exc}")
    else:
        raise SystemExit("ownership loss incorrectly reported a restored clipboard")
    if fake.text != newer:
        raise SystemExit("ownership loss overwrote the newer user clipboard")
    conditional_calls = [
        argv
        for argv, _kwargs in fake.calls
        if cs._CONDITIONAL_PLAIN_TEXT_WRITE_JXA in argv
    ]
    if len(conditional_calls) != 1:
        raise SystemExit(
            f"ownership loss must skip the restore write: {conditional_calls}"
        )


def test_restage_preserves_original_snapshot_and_updates_ownership() -> None:
    original = "original user clipboard"
    recipient = "synthetic recipient"
    message = "synthetic message"
    fake = FakePasteboardRun(original)
    with cs.PlainTextClipboardTransaction(run=fake) as transaction:
        transaction.stage(recipient)
        first_owned_count = fake.change_count
        transaction.restage(message)
        if fake.text != message or fake.change_count == first_owned_count:
            raise SystemExit("clipboard restage did not replace the owned staged value")
        if original in repr(transaction) or recipient in repr(transaction) or message in repr(transaction):
            raise SystemExit("clipboard restage leaked private content through repr")
    if fake.text != original or transaction.restored is not True:
        raise SystemExit("clipboard restage did not restore the original snapshot")
    writes = [
        kwargs.get("input")
        for argv, kwargs in fake.calls
        if cs._CONDITIONAL_PLAIN_TEXT_WRITE_JXA in argv
    ]
    if writes != [recipient, message, original]:
        raise SystemExit(f"clipboard restage write order drifted: {writes}")
    _assert_no_content_in_argv(fake, original, recipient, message)


def test_restage_ownership_loss_preserves_newer_clipboard() -> None:
    original = "original user clipboard"
    recipient = "synthetic recipient"
    newer = "newer application clipboard"
    fake = FakePasteboardRun(original)
    transaction = cs.PlainTextClipboardTransaction(run=fake)
    transaction.__enter__()
    transaction.stage(recipient)
    fake.external_copy(newer)
    try:
        transaction.restage("synthetic message")
    except cs.ClipboardSafetyError as exc:
        if exc.stage != "clipboard_ownership_lost":
            raise SystemExit(f"clipboard restage ownership stage drifted: {exc.stage}")
    else:
        raise SystemExit("clipboard restage accepted lost ownership")
    transaction.__exit__(None, None, None)
    if fake.text != newer:
        raise SystemExit("clipboard restage overwrote a newer clipboard")
    writes = [
        argv
        for argv, _kwargs in fake.calls
        if cs._CONDITIONAL_PLAIN_TEXT_WRITE_JXA in argv
    ]
    if len(writes) != 1:
        raise SystemExit("lost restage ownership attempted a replacement or restore")
    privacy = transaction.privacy_metadata()
    if (
        privacy.get("clipboard_text_restored") is not False
        or privacy.get("clipboard_private_content_possible") is not True
    ):
        raise SystemExit(f"lost restage ownership overclaimed custody: {privacy}")


def test_conditional_restage_failure_never_guesses_ownership() -> None:
    fake = FakePasteboardRun("original")
    transaction = cs.PlainTextClipboardTransaction(run=fake)
    transaction.__enter__()
    transaction.stage("recipient")
    fake.stale_next_write = True
    try:
        transaction.restage("message")
    except cs.ClipboardSafetyError as exc:
        if exc.stage != "clipboard_ownership_lost":
            raise SystemExit(f"conditional restage failure stage drifted: {exc.stage}")
    else:
        raise SystemExit("conditional restage failure guessed clipboard ownership")
    transaction.__exit__(None, None, None)
    if fake.text != "recipient":
        raise SystemExit("conditional restage failure performed an unsafe restore")
    writes = [
        argv
        for argv, _kwargs in fake.calls
        if cs._CONDITIONAL_PLAIN_TEXT_WRITE_JXA in argv
    ]
    if len(writes) != 2:
        raise SystemExit(f"conditional restage failure performed extra writes: {writes}")


def test_conditional_stage_guard_preserves_newer_clipboard() -> None:
    original = "old user clipboard"
    staged = "private staged message"
    fake = FakePasteboardRun(original)
    transaction = cs.PlainTextClipboardTransaction(run=fake)
    transaction.__enter__()
    # Equal text is adversarial: a stale helper response proves Jarvis did not
    # write it, so equality must not be mistaken for transaction ownership.
    fake.external_copy(staged)
    fake.stale_next_write = True
    try:
        transaction.stage(staged)
    except cs.ClipboardSafetyError as exc:
        if exc.stage != "clipboard_ownership_lost":
            raise SystemExit(f"conditional stage guard drifted: {exc.stage}")
    else:
        raise SystemExit("conditional stage guard accepted stale ownership")
    try:
        transaction.__exit__(None, None, None)
    except cs.ClipboardSafetyError as exc:
        if exc.stage != "clipboard_ownership_lost":
            raise
    if fake.text != staged:
        raise SystemExit("conditional stage guard overwrote newer clipboard")
    conditional_calls = [
        argv
        for argv, _kwargs in fake.calls
        if cs._CONDITIONAL_PLAIN_TEXT_WRITE_JXA in argv
    ]
    if len(conditional_calls) != 1:
        raise SystemExit("stale conditional stage must not attempt restoration")


def test_action_error_restores_then_reraises_action() -> None:
    fake = FakePasteboardRun("original")
    try:
        with cs.temporary_plain_text_clipboard("staged", run=fake):
            raise RuntimeError("action failed")
    except RuntimeError as exc:
        if str(exc) != "action failed":
            raise
    else:
        raise SystemExit("action exception was swallowed")
    if fake.text != "original":
        raise SystemExit("action exception did not restore the clipboard")


def test_restore_verification_fails_closed_without_content_leak() -> None:
    original = "PRIVATE original"
    staged = "PRIVATE staged"
    fake = FakePasteboardRun(original)
    fake.corrupt_after_write_number = 2
    try:
        with cs.temporary_plain_text_clipboard(staged, run=fake):
            pass
    except cs.ClipboardSafetyError as exc:
        if exc.stage != "clipboard_restore_verification":
            raise SystemExit(f"verification failure stage drifted: {exc.stage}")
        if "PRIVATE" in str(exc):
            raise SystemExit(f"verification failure leaked content: {exc}")
    else:
        raise SystemExit("corrupt clipboard verification reported success")


def test_static_failures_do_not_surface_stderr_or_input() -> None:
    fake = FakePasteboardRun("original")
    fake.fail_next_write = True
    try:
        with cs.temporary_plain_text_clipboard(
            "PRIVATE /\x55sers/example/staged.txt",
            run=fake,
        ):
            raise AssertionError("failed stage entered transaction body")
    except cs.ClipboardSafetyError as exc:
        if exc.stage != "clipboard_write":
            raise SystemExit(f"write failure stage drifted: {exc.stage}")
        if "PRIVATE" in str(exc) or "/\x55sers/" in str(exc):
            raise SystemExit(f"write failure leaked private detail: {exc}")
    else:
        raise SystemExit("failed conditional write reported success")
    _assert_no_content_in_argv(
        fake,
        "original",
        "PRIVATE /\x55sers/example/staged.txt",
    )


def main() -> None:
    test_happy_path_restores_exact_text_without_argv_leak()
    test_rich_clipboard_refuses_before_write()
    test_snapshot_change_fails_before_write()
    test_ownership_loss_skips_restore_and_preserves_newer_clipboard()
    test_restage_preserves_original_snapshot_and_updates_ownership()
    test_restage_ownership_loss_preserves_newer_clipboard()
    test_conditional_restage_failure_never_guesses_ownership()
    test_conditional_stage_guard_preserves_newer_clipboard()
    test_action_error_restores_then_reraises_action()
    test_restore_verification_fails_closed_without_content_leak()
    test_static_failures_do_not_surface_stderr_or_input()
    print("Clipboard safety smoke passed")


if __name__ == "__main__":
    main()
