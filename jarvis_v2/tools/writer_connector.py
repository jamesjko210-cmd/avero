"""Human-style writing for Jarvis V2 (types into any focused app via System Events).

Two capabilities:
  - paste_text:  drop a block of text instantly (clipboard paste)
  - human_write: type character-by-character with human realism — variable
                 timing, muscle-memory bursts, sentence-end pauses, occasional
                 hesitations, and ~1.4% typos that get noticed and corrected.

Works anywhere a cursor can go (Google Docs, Sheets, Slides, editors, forms)
because it drives macOS System Events key codes. Both are HIGH_RISK (they
control the keyboard) and stay approval-gated.

Requires Accessibility permission for the Python interpreter:
  System Settings → Privacy & Security → Accessibility.
"""

from __future__ import annotations

import random
import re
import subprocess
import time
from typing import Any

from jarvis_v2.agent.failure_guidance import declare_outcome_unknown_failure
from jarvis_v2.agent.types import RiskLevel, ToolResult
from jarvis_v2.config import JarvisConfig
from jarvis_v2.tools import clipboard_safety


MAX_TEXT_CHARS = 20000
DEFAULT_COUNTDOWN_SECONDS = 3
MAX_COUNTDOWN_SECONDS = 15
MAX_RAW_COUNTDOWN_CHARS = 80
LOCAL_PATH_RE = re.compile(r"/(?:Users|private|var/folders|tmp)/[^\n\r;]*")

SPEEDS = {
    "slow": (165, 52),
    "relaxed": (100, 34),
    "normal": (88, 30),
    "fast": (46, 18),
    "blazing": (18, 8),
}
FAST_BIGRAMS = {
    "th", "he", "in", "er", "an", "re", "on", "en", "at", "es", "ed", "nd", "to",
    "or", "it", "is", "ar", "te", "ti", "le", "st", "ng", "co", "ha", "se", "ou",
    "ea", "de", "al", "hi", "io", "nt", "ve", "ll", "si", "ow", "ro", "me", "pe",
    "ri", "ca", "ra", "ss", "ee", "oo", "tt", "ff", "pp", "cc", "mm",
}
BURST_WORDS = {
    "the", "and", "is", "it", "a", "an", "to", "of", "in", "for", "on", "at", "by",
    "or", "as", "be", "we", "he", "she", "they", "his", "her", "its", "our", "you",
    "i", "me", "my", "us", "so", "but", "not", "if", "do", "go", "up", "no", "was",
    "are", "has", "had", "did", "can", "will", "all", "one", "two", "this", "that",
    "with", "from", "into", "just", "been", "also", "more", "than", "then", "when",
    "what", "who", "how", "out", "now", "new", "any", "get", "got", "see", "said",
    "say", "may", "some", "very", "each",
}
ADJACENT: dict[str, str] = {
    "q": "wa", "w": "qesa", "e": "wrds", "r": "etfd", "t": "ryfg", "y": "tugh",
    "u": "yihj", "i": "uojk", "o": "ipkl", "p": "ol", "a": "qsz", "s": "awdxz",
    "d": "sefcx", "f": "drgvc", "g": "ftbhv", "h": "gynbj", "j": "humkn",
    "k": "jilm", "l": "kop", "z": "ax", "x": "zsc", "c": "xdv", "v": "cfb",
    "b": "vgn", "n": "bhm", "m": "nj",
}
TYPO_RATE = 0.014


def _safe_metadata(**extra: Any) -> dict[str, Any]:
    base = {
        "calls_model": False,
        "calls_external_service": False,
        "executes_tools": False,
        "reads_personal_data": False,
        "reads_private_data": False,
        "executes_side_effect": True,
        "external_side_effect": False,
        "writes_files": False,
        "writes_database": False,
        "writes_memory": False,
        "writes_notes": False,
        "queues_approval": False,
        "requires_approval": True,
        "controls_computer": True,
        "authorizes_execution": False,
        "authorizes_completion_claim": False,
        "approval_granted": False,
    }
    base.update(extra)
    return base


def _writer_boundaries(
    *,
    controls_computer: bool = False,
    reads_clipboard: bool = False,
    executes_side_effect: bool | None = None,
    requires_approval: bool | None = None,
) -> dict[str, bool]:
    if executes_side_effect is None:
        executes_side_effect = controls_computer
    if requires_approval is None:
        requires_approval = controls_computer or reads_clipboard
    return {
        "calls_model": False,
        "calls_external_service": False,
        "executes_tools": False,
        "reads_personal_data": reads_clipboard,
        "reads_private_data": reads_clipboard,
        "reads_clipboard": reads_clipboard,
        "executes_side_effect": executes_side_effect,
        "external_side_effect": False,
        "writes_files": False,
        "writes_database": False,
        "writes_memory": False,
        "writes_notes": False,
        "queues_approval": False,
        "requires_approval": requires_approval,
        "controls_computer": controls_computer,
        "authorizes_execution": False,
        "authorizes_completion_claim": False,
        "approval_granted": False,
    }


def _writer_handoff(
    *,
    tool_name: str,
    status: str,
    reason: str = "",
    text: str = "",
    countdown: int | None = None,
    raw_countdown: str | None = None,
    speed: str = "",
    typos: bool | None = None,
    exception_type: str = "",
    error_type: str = "",
    controls_computer: bool = False,
    failure_stage: str = "",
    paste_command_completed: bool = False,
    clipboard_restored: bool | None = None,
    clipboard_text_restored: bool | None = None,
    clipboard_fully_restored: bool | None = None,
    clipboard_restoration_scope: str = "",
    clipboard_outcome_known: bool = True,
    clipboard_snapshot_attempted: bool | None = False,
    clipboard_snapshot_captured: bool | None = False,
    clipboard_replacement_attempted: bool | None = False,
    clipboard_private_content_possible: bool = False,
    reads_clipboard: bool = False,
    executes_side_effect: bool | None = None,
    requires_approval: bool | None = None,
    target_content_verified: bool = False,
    retry_safe: bool = False,
) -> dict[str, Any]:
    if status == "outcome_unknown":
        next_safe_command = "recent tool runs"
        next_safe_commands = ["recent tool runs", "execution recovery"]
    elif status == "failed":
        next_safe_command = "setup check"
        next_safe_commands = ["setup check"]
    elif status in {"written", "typing_command_completed", "paste_command_completed"}:
        next_safe_command = "recent tool runs"
        next_safe_commands = ["recent tool runs"]
    else:
        next_safe_command = "safe next actions"
        next_safe_commands = ["safe next actions"]
    return {
        "writer_handoff": {
            "source": tool_name,
            "status": status,
            "reason": reason,
            "chars": len(text),
            "local_path_text": bool(LOCAL_PATH_RE.search(text or "")),
            "text_content_in_metadata": False,
            "countdown": countdown,
            "raw_countdown": raw_countdown,
            "speed": speed,
            "typos": typos,
            "exception_type": exception_type,
            "error_type": error_type,
            "failure_stage": failure_stage,
            "paste_command_completed": paste_command_completed,
            "clipboard_restored": clipboard_restored,
            "clipboard_text_restored": clipboard_text_restored,
            "clipboard_fully_restored": clipboard_fully_restored,
            "clipboard_restoration_scope": clipboard_restoration_scope,
            "clipboard_outcome_known": clipboard_outcome_known,
            "clipboard_snapshot_attempted": clipboard_snapshot_attempted,
            "clipboard_snapshot_captured": clipboard_snapshot_captured,
            "clipboard_replacement_attempted": clipboard_replacement_attempted,
            "clipboard_private_content_possible": clipboard_private_content_possible,
            "target_content_verified": target_content_verified,
            "retry_safe": retry_safe,
            "approval_required_before_execution": (
                requires_approval
                if requires_approval is not None
                else controls_computer or reads_clipboard
            ),
            "manual_review_required": (
                requires_approval
                if requires_approval is not None
                else controls_computer or reads_clipboard
            ),
            "authorizes_execution": False,
            "authorizes_completion_claim": False,
            "approval_granted": False,
            "next_safe_command": next_safe_command,
            "next_safe_commands": next_safe_commands,
            "boundaries": _writer_boundaries(
                controls_computer=controls_computer,
                reads_clipboard=reads_clipboard,
                executes_side_effect=executes_side_effect,
                requires_approval=requires_approval,
            ),
        }
    }


def _short_raw(value: Any, *, limit: int = MAX_RAW_COUNTDOWN_CHARS) -> str:
    text = str(value)
    text = LOCAL_PATH_RE.sub("<local-path>", text)
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "…"


def _looks_like_local_path(value: str) -> bool:
    return bool(LOCAL_PATH_RE.search(value))


def _countdown_metadata(value: Any, *, sanitized: int) -> dict[str, Any]:
    if value is None or value == "":
        return {"countdown": sanitized}
    if isinstance(value, bool):
        return {"countdown": sanitized, "raw_countdown": str(value)}
    try:
        int(value)
    except (TypeError, ValueError):
        return {"countdown": sanitized, "raw_countdown": _short_raw(value)}
    return {"countdown": sanitized}


def _raw_countdown_display(value: Any) -> str | None:
    return _countdown_metadata(value, sanitized=DEFAULT_COUNTDOWN_SECONDS).get("raw_countdown")


def _bounded_countdown(value: Any) -> int:
    if value is None or value == "" or isinstance(value, bool):
        return DEFAULT_COUNTDOWN_SECONDS
    try:
        seconds = int(value)
    except (TypeError, ValueError):
        seconds = DEFAULT_COUNTDOWN_SECONDS
    return max(0, min(seconds, MAX_COUNTDOWN_SECONDS))


def _write_failure_message(mode: str) -> str:
    recovery = (
        "Keep the target app focused, grant Accessibility access to Jarvis/Python/Terminal in "
        "System Settings > Privacy & Security > Accessibility, run `setup check`, then retry "
        "after approval."
    )
    if mode == "paste":
        return f"I couldn't paste into the active window. {recovery}"
    return f"I couldn't type into the active window. {recovery}"


class _PasteBlockError(RuntimeError):
    def __init__(
        self,
        stage: str,
        *,
        paste_command_completed: bool = False,
        paste_outcome_possible: bool = False,
        clipboard_restored: bool | None = None,
        retry_safe: bool = False,
        clipboard_privacy: dict[str, Any] | None = None,
    ):
        super().__init__(stage)
        self.stage = stage
        self.paste_command_completed = paste_command_completed
        self.paste_outcome_possible = paste_outcome_possible
        self.clipboard_restored = clipboard_restored
        self.retry_safe = retry_safe
        self.clipboard_privacy = dict(clipboard_privacy or {})


def _paste_failure_message(error: _PasteBlockError) -> str:
    if error.stage == "clipboard_rich_content":
        return (
            "Jarvis stopped before changing the clipboard because it contains rich or non-text "
            "data. Copy plain text or clear the clipboard, keep the target focused, then submit "
            "a new approved paste."
        )
    if error.stage == "clipboard_inspection":
        return (
            "Jarvis could not verify that the clipboard is plain text, so it stopped before "
            "changing anything. Run `setup check`, repair clipboard access, then submit a new "
            "approved paste."
        )
    if error.paste_outcome_possible or error.paste_command_completed:
        clipboard_note = (
            " The previous clipboard content may not have been restored."
            if error.clipboard_restored is False
            else ""
        )
        return (
            "The paste outcome is uncertain; text may already be in the active window."
            f"{clipboard_note} Check the target and clipboard before deciding whether to retry."
        )
    return _write_failure_message("paste")


def _post_attempt_writer_recovery_guidance(mode: str) -> str:
    if mode == "paste":
        return (
            "The paste outcome is unknown; text may already be in the active window. "
            "Check the target and clipboard, and do not paste again automatically. "
            "If absent, run `setup check`, repair focus or Accessibility, then submit a new approved paste."
        )
    if mode == "clipboard":
        return (
            "The target was not pasted, but clipboard restoration is uncertain. Check the "
            "clipboard and do not repeat automatically. Run `setup check`, repair clipboard "
            "access, then submit a new approved paste."
        )
    return (
        "The typing outcome is unknown; some text may already be in the active window. "
        "Check the target and do not type it again automatically. If incomplete, run "
        "`setup check`, repair focus or Accessibility, then submit a new approved write."
    )


def _char_delay(char: str, prev: str | None, mean: float, std: float) -> float:
    ms = max(mean * 0.18, random.gauss(mean, std))
    if char.isupper():
        ms *= random.uniform(1.12, 1.55)
    if char in ".,;:!?":
        ms *= random.uniform(1.10, 1.75)
    if char in r"""()[]{}@#$%^&*_=+|\/<>~`"'""":
        ms *= random.uniform(1.35, 2.25)
    if prev and (prev + char).lower() in FAST_BIGRAMS:
        ms *= random.uniform(0.40, 0.72)
    if char == " ":
        ms *= random.uniform(0.75, 1.25)
    if char == "\n":
        ms = max(mean * 1.5, random.gauss(mean * 4.5, mean * 1.2))
    return ms


def _extra_delay(char: str, nxt: str | None, mean: float) -> float:
    extra = 0.0
    if char in ".!?" and nxt in (" ", "\n", None):
        extra += random.uniform(mean * 1.5, mean * 5.5)
    roll = random.random()
    if roll < 0.004:
        extra += random.uniform(1200, 3400)
    elif roll < 0.020:
        extra += random.uniform(350, 1100)
    return extra


def plan_keystrokes(text: str, mean: float, std: float, typos: bool) -> list[tuple]:
    actions: list[tuple] = []
    prev: str | None = None
    burst_left = 0
    burst_mult = 1.0
    i = 0
    while i < len(text):
        char = text[i]
        nxt = text[i + 1] if i + 1 < len(text) else None
        if char.isalpha() and (prev is None or not prev.isalpha()):
            j = i
            while j < len(text) and text[j].isalpha():
                j += 1
            word = text[i:j].lower()
            if word in BURST_WORDS:
                burst_left = j - i
                burst_mult = random.uniform(0.50, 0.78)
            else:
                burst_left = 0
                burst_mult = 1.0
        ms = _char_delay(char, prev, mean, std) + _extra_delay(char, nxt, mean)
        if burst_left > 0:
            ms *= burst_mult
            burst_left -= 1
        if typos and char.islower() and char in ADJACENT and random.random() < TYPO_RATE:
            wrong = random.choice(ADJACENT[char])
            actions.append(("char", wrong, ms))
            actions.append(("backspace", None, random.uniform(500, 950)))
            actions.append(("char", char, random.uniform(200, 420)))
        elif char == "\n":
            actions.append(("enter", None, ms))
        elif char == "\t":
            actions.append(("tab", None, ms))
        else:
            actions.append(("char", char, ms))
        prev = char
        i += 1
    return actions


def build_script(actions: list[tuple]) -> str:
    lines = ['tell application "System Events"']
    for kind, payload, delay_ms in actions:
        lines.append(f"    delay {delay_ms / 1000.0:.5f}")
        if kind == "char":
            lines.append(f"    keystroke (character id {ord(payload)})")
        elif kind == "enter":
            lines.append("    key code 36")
        elif kind == "backspace":
            lines.append("    key code 51 using {}")
        elif kind == "tab":
            lines.append("    key code 48")
    lines.append("end tell")
    return "\n".join(lines)


def _run_actions(actions: list[tuple], chunk: int = 400) -> None:
    for start in range(0, len(actions), chunk):
        script = build_script(actions[start:start + chunk])
        result = subprocess.run(["osascript", "-e", script], capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            raise RuntimeError((result.stderr or result.stdout or "AppleScript failed").strip())


def _paste_block(text: str) -> dict[str, Any]:
    transaction = clipboard_safety.PlainTextClipboardTransaction(
        run=subprocess.run,
    )
    paste_attempted = False
    paste_completed = False
    try:
        with transaction:
            transaction.stage(text)
            paste_attempted = True
            pasted = subprocess.run(
                [
                    "osascript",
                    "-e",
                    'tell application "System Events" to keystroke "v" using command down',
                ],
                capture_output=True,
                text=True,
                timeout=15,
            )
            if pasted.returncode != 0:
                raise RuntimeError("paste_command")
            paste_completed = True
            time.sleep(0.3)
    except clipboard_safety.ClipboardSafetyError as exc:
        privacy = transaction.privacy_metadata()
        replacement_attempted = privacy.get("clipboard_replacement_attempted") is True
        text_restored = privacy.get("clipboard_text_restored")
        raise _PasteBlockError(
            exc.stage,
            paste_command_completed=paste_completed,
            paste_outcome_possible=paste_attempted,
            clipboard_restored=(
                None
                if replacement_attempted and text_restored is True
                else False
                if replacement_attempted
                else True
            ),
            retry_safe=(
                not paste_attempted
                and privacy.get("clipboard_outcome_known") is True
            ),
            clipboard_privacy=privacy,
        ) from exc
    except Exception as exc:
        privacy = transaction.privacy_metadata()
        replacement_attempted = privacy.get("clipboard_replacement_attempted") is True
        text_restored = privacy.get("clipboard_text_restored")
        raise _PasteBlockError(
            "paste_command" if paste_attempted else "clipboard_write",
            paste_command_completed=paste_completed,
            paste_outcome_possible=paste_attempted,
            clipboard_restored=(
                None
                if replacement_attempted and text_restored is True
                else False
                if replacement_attempted
                else True
            ),
            retry_safe=(
                not paste_attempted
                and privacy.get("clipboard_outcome_known") is True
            ),
            clipboard_privacy=privacy,
        ) from exc
    return transaction.privacy_metadata()


def make_writer_tools(config: JarvisConfig):
    def paste_text(args: dict[str, Any]) -> ToolResult:
        text = str(args.get("text") or "")[:MAX_TEXT_CHARS]
        if not text.strip():
            return ToolResult(
                "paste_text",
                False,
                "Give me the text to write.",
                _safe_metadata(
                    executes_side_effect=False,
                    requires_approval=False,
                    controls_computer=False,
                    reason="missing_text",
                    **_writer_handoff(tool_name="paste_text", status="refused", reason="missing_text"),
                ),
            )
        if _looks_like_local_path(text):
            return ToolResult(
                "paste_text",
                False,
                "Please give me text to write, not a local file path.",
                _safe_metadata(
                    chars=len(text),
                    reason="invalid_text",
                    local_path_text=True,
                    executes_side_effect=False,
                    requires_approval=False,
                    controls_computer=False,
                    **_writer_handoff(tool_name="paste_text", status="refused", reason="invalid_text", text=text),
                ),
            )
        countdown = _bounded_countdown(args.get("countdown"))
        countdown_meta = _countdown_metadata(args.get("countdown"), sanitized=countdown)
        paste_started = False
        try:
            time.sleep(countdown)  # let the user focus the target window
            paste_started = True
            clipboard_privacy = _paste_block(text)
            return ToolResult(
                "paste_text",
                True,
                (
                    f"Paste command completed for {len(text)} characters; confirm the target. "
                    "The exact prior plain-text clipboard value was restored; its original "
                    "pasteboard representation set was not preserved."
                ),
                _safe_metadata(
                    chars=len(text),
                    reads_personal_data=True,
                    reads_private_data=True,
                    reads_clipboard=True,
                    paste_command_completed=True,
                    clipboard_restored=None,
                    **clipboard_privacy,
                    target_content_verified=False,
                    retry_safe=False,
                    **countdown_meta,
                    **_writer_handoff(
                        tool_name="paste_text",
                        status="paste_command_completed",
                        text=text,
                        countdown=countdown,
                        raw_countdown=countdown_meta.get("raw_countdown"),
                        controls_computer=True,
                        reads_clipboard=True,
                        executes_side_effect=True,
                        requires_approval=True,
                        paste_command_completed=True,
                        clipboard_restored=None,
                        clipboard_text_restored=clipboard_privacy.get("clipboard_text_restored"),
                        clipboard_fully_restored=clipboard_privacy.get("clipboard_fully_restored"),
                        clipboard_restoration_scope=str(
                            clipboard_privacy.get("clipboard_restoration_scope") or ""
                        ),
                        clipboard_outcome_known=clipboard_privacy.get("clipboard_outcome_known") is True,
                        clipboard_snapshot_attempted=clipboard_privacy.get("clipboard_snapshot_attempted"),
                        clipboard_snapshot_captured=clipboard_privacy.get("clipboard_snapshot_captured"),
                        clipboard_replacement_attempted=clipboard_privacy.get("clipboard_replacement_attempted"),
                        clipboard_private_content_possible=clipboard_privacy.get("clipboard_private_content_possible") is True,
                        target_content_verified=False,
                        retry_safe=False,
                    ),
                ),
            )
        except Exception as e:
            paste_error = e if isinstance(e, _PasteBlockError) else None
            clipboard_privacy = (
                dict(paste_error.clipboard_privacy)
                if paste_error is not None and paste_error.clipboard_privacy
                else clipboard_safety.privacy_boundary_metadata(
                    snapshot_attempted=paste_started,
                    snapshot_captured=False,
                    replacement_attempted=paste_started,
                    text_restored=False if paste_started else None,
                )
            )
            clipboard_unknown = (
                clipboard_privacy.get("clipboard_outcome_known") is not True
            )
            clipboard_only_unknown = bool(
                clipboard_unknown
                and paste_error
                and not paste_error.paste_outcome_possible
            )
            raw_clipboard_restored = (
                paste_error.clipboard_restored if paste_error is not None else None
            )
            outcome_unknown = bool(
                clipboard_unknown
                or (paste_error and paste_error.paste_outcome_possible)
                or (paste_started and paste_error is None)
            )
            rich_clipboard_refusal = bool(
                paste_error and paste_error.stage == "clipboard_rich_content"
            )
            failure_output = (
                _post_attempt_writer_recovery_guidance(
                    "clipboard" if clipboard_only_unknown else "paste"
                )
                if outcome_unknown
                else _paste_failure_message(paste_error)
                if paste_error is not None
                else _write_failure_message("paste")
            )
            reads_clipboard = (
                clipboard_privacy.get("clipboard_snapshot_attempted") is True
            )
            controls_computer = bool(
                (paste_error and paste_error.paste_outcome_possible)
                or (paste_error and paste_error.paste_command_completed)
                or (paste_started and paste_error is None)
            )
            executes_side_effect = bool(
                clipboard_privacy.get("clipboard_replacement_attempted") is True
                or controls_computer
            )
            failure_metadata = _safe_metadata(
                reads_personal_data=reads_clipboard,
                reads_private_data=reads_clipboard,
                reads_clipboard=reads_clipboard,
                executes_side_effect=executes_side_effect,
                requires_approval=reads_clipboard or controls_computer,
                controls_computer=controls_computer,
                exception_type=type(e).__name__,
                error_type=type(e).__name__,
                failure_stage=paste_error.stage if paste_error is not None else "paste_unknown",
                paste_command_completed=bool(paste_error and paste_error.paste_command_completed),
                paste_outcome_possible=bool(paste_error and paste_error.paste_outcome_possible),
                clipboard_restored=raw_clipboard_restored,
                **clipboard_privacy,
                target_content_verified=False,
                outcome_known=not outcome_unknown,
                outcome_unknown=outcome_unknown,
                side_effect_possible=outcome_unknown,
                retry_safe=bool(paste_error and paste_error.retry_safe) if not outcome_unknown else False,
                automatic_retry_allowed=False,
                authorizes_retry=False,
                **countdown_meta,
                **_writer_handoff(
                    tool_name="paste_text",
                    status=(
                        "outcome_unknown"
                        if outcome_unknown
                        else "refused"
                        if rich_clipboard_refusal
                        else "failed"
                    ),
                    reason=(
                        "clipboard_outcome_unknown"
                        if clipboard_only_unknown
                        else "paste_outcome_unknown"
                        if outcome_unknown
                        else "clipboard_requires_plain_text"
                        if rich_clipboard_refusal
                        else "write_error"
                    ),
                    text=text,
                    countdown=countdown,
                    raw_countdown=countdown_meta.get("raw_countdown"),
                    exception_type=type(e).__name__,
                    error_type=type(e).__name__,
                    controls_computer=controls_computer,
                    reads_clipboard=reads_clipboard,
                    executes_side_effect=executes_side_effect,
                    requires_approval=reads_clipboard or controls_computer,
                    failure_stage=paste_error.stage if paste_error is not None else "paste_unknown",
                    paste_command_completed=bool(paste_error and paste_error.paste_command_completed),
                    clipboard_restored=raw_clipboard_restored,
                    clipboard_text_restored=clipboard_privacy.get("clipboard_text_restored"),
                    clipboard_fully_restored=clipboard_privacy.get("clipboard_fully_restored"),
                    clipboard_restoration_scope=str(
                        clipboard_privacy.get("clipboard_restoration_scope") or ""
                    ),
                    clipboard_outcome_known=not clipboard_unknown,
                    clipboard_snapshot_attempted=clipboard_privacy.get("clipboard_snapshot_attempted"),
                    clipboard_snapshot_captured=clipboard_privacy.get("clipboard_snapshot_captured"),
                    clipboard_replacement_attempted=clipboard_privacy.get("clipboard_replacement_attempted"),
                    clipboard_private_content_possible=clipboard_privacy.get("clipboard_private_content_possible") is True,
                    target_content_verified=False,
                    retry_safe=bool(paste_error and paste_error.retry_safe) if not outcome_unknown else False,
                ),
            )
            if outcome_unknown:
                failure_metadata = declare_outcome_unknown_failure(
                    failure_metadata,
                    output=failure_output,
                    commands=("setup check",),
                )
            return ToolResult(
                "paste_text",
                False,
                failure_output,
                failure_metadata,
            )

    def human_write(args: dict[str, Any]) -> ToolResult:
        text = str(args.get("text") or "")[:MAX_TEXT_CHARS]
        if not text.strip():
            return ToolResult(
                "human_write",
                False,
                "Give me the text to type out.",
                _safe_metadata(
                    executes_side_effect=False,
                    requires_approval=False,
                    controls_computer=False,
                    reason="missing_text",
                    **_writer_handoff(tool_name="human_write", status="refused", reason="missing_text"),
                ),
            )
        if _looks_like_local_path(text):
            return ToolResult(
                "human_write",
                False,
                "Please give me text to type, not a local file path.",
                _safe_metadata(
                    chars=len(text),
                    reason="invalid_text",
                    local_path_text=True,
                    executes_side_effect=False,
                    requires_approval=False,
                    controls_computer=False,
                    **_writer_handoff(tool_name="human_write", status="refused", reason="invalid_text", text=text),
                ),
            )
        speed = str(args.get("speed") or "normal").strip().lower()
        if speed not in SPEEDS:
            speed = "normal"
        typos = str(args.get("typos", "true")).strip().lower() not in {"false", "no", "off", "0"}
        countdown = _bounded_countdown(args.get("countdown"))
        countdown_meta = _countdown_metadata(args.get("countdown"), sanitized=countdown)
        mean, std = SPEEDS[speed]
        write_started = False
        try:
            time.sleep(countdown)  # let the user focus the target window
            actions = plan_keystrokes(text, mean, std, typos)
            write_started = True
            _run_actions(actions)
            return ToolResult(
                "human_write", True,
                (
                    f"Typing command completed for {len(text)} characters "
                    f"(speed={speed}, typos={'on' if typos else 'off'}); confirm the target."
                ),
                _safe_metadata(
                    chars=len(text),
                    speed=speed,
                    typos=typos,
                    write_attempted=True,
                    target_content_verified=False,
                    **countdown_meta,
                    **_writer_handoff(
                        tool_name="human_write",
                        status="typing_command_completed",
                        text=text,
                        countdown=countdown,
                        raw_countdown=countdown_meta.get("raw_countdown"),
                        speed=speed,
                        typos=typos,
                        controls_computer=True,
                    ),
                ),
            )
        except Exception as e:
            outcome_unknown = write_started
            failure_output = (
                _post_attempt_writer_recovery_guidance("human")
                if outcome_unknown
                else _write_failure_message("human")
            )
            failure_metadata = _safe_metadata(
                executes_side_effect=write_started,
                requires_approval=write_started,
                controls_computer=write_started,
                exception_type=type(e).__name__,
                error_type=type(e).__name__,
                failure_stage="typing_actions" if outcome_unknown else "typing_preflight",
                write_attempted=write_started,
                target_content_verified=False,
                outcome_known=not outcome_unknown,
                outcome_unknown=outcome_unknown,
                side_effect_possible=outcome_unknown,
                retry_safe=not outcome_unknown,
                automatic_retry_allowed=False,
                authorizes_retry=False,
                **countdown_meta,
                **_writer_handoff(
                    tool_name="human_write",
                    status="outcome_unknown" if outcome_unknown else "failed",
                    reason="write_outcome_unknown" if outcome_unknown else "write_error",
                    text=text,
                    countdown=countdown,
                    raw_countdown=countdown_meta.get("raw_countdown"),
                    speed=speed,
                    typos=typos,
                    exception_type=type(e).__name__,
                    error_type=type(e).__name__,
                    controls_computer=write_started,
                    failure_stage="typing_actions" if outcome_unknown else "typing_preflight",
                    target_content_verified=False,
                    retry_safe=not outcome_unknown,
                ),
            )
            if outcome_unknown:
                failure_metadata = declare_outcome_unknown_failure(
                    failure_metadata,
                    output=failure_output,
                    commands=("setup check",),
                )
            return ToolResult(
                "human_write",
                False,
                failure_output,
                failure_metadata,
            )

    from jarvis_v2.tools.registry import Tool
    return [
        Tool(
            "paste_text",
            "Instantly write a block of text into the focused window (Google Docs, Sheets, anywhere). Args: text, countdown (secs to switch windows).",
            RiskLevel.HIGH_RISK,
            paste_text,
            "personal",
        ),
        Tool(
            "human_write",
            "Type text into the focused window like a human — realistic timing, bursts, pauses, and occasional self-corrected typos. Args: text, speed (slow|normal|fast|blazing), typos (true/false), countdown.",
            RiskLevel.HIGH_RISK,
            human_write,
            "personal",
        ),
    ]
