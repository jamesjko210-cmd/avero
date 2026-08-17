"""Jarvis HUD — a small always-on-top window for talking to Jarvis by voice.

This is the COMPACT companion to the full Codex-built Jarvis dashboard: same
runtime, same approval gates, just a minimal floating window.

Built for WhisperFlow (or any dictation tool). The window is native AppKit
(via PyObjC): dictation tools insert text through macOS accessibility APIs,
which see native NSTextField controls but NOT Tk widgets — the original Tk
input accepted typing but was invisible to WhisperFlow. A Tk fallback remains
for machines without PyObjC (typing-only there).

Every command runs through the SAME JarvisRuntime as Telegram — planner,
tools, approval gates, audit log. Replies are shown on screen and (optionally)
spoken with macOS `say`; the ⏹ button (or toggling 🔊 off) silences speech
that is already playing — no need to sit through a long reply.

    JARVIS_V3_ENV="$HOME/.jarvis_v3/runtime.env" .venv/bin/python3 -m jarvis_v2.scripts.hud
    JARVIS_V3_ENV="$HOME/.jarvis_v3/runtime.env" JARVIS_HUD_SPEAK=0 .venv/bin/python3 -m jarvis_v2.scripts.hud

The optional com.jarvis-v3.hud LaunchAgent can start it at login after supervised cutover.
"""

from __future__ import annotations

import os
import queue
import subprocess
import tempfile
import threading
from dataclasses import dataclass, field
from pathlib import Path

from jarvis_v2.scripts.chat import _approval_ids_from_result
from jarvis_v2.v3_commands import V3_BOOTSTRAP_CHECK_COMMAND


@dataclass
class HudReply:
    """One turn's outcome, UI-toolkit-agnostic (testable without a window)."""

    response: str
    approval_ids: list[int] = field(default_factory=list)
    error: str = ""


class HudCore:
    """Runtime wrapper for the HUD: lazy init, one command at a time.

    Kept free of any UI imports so the logic is smoke-testable headlessly.
    `runtime_factory` is injectable for tests; the default builds the real
    JarvisRuntime on first use (keeps window startup instant).
    """

    def __init__(self, runtime_factory=None) -> None:
        self._runtime = None
        self._runtime_factory = runtime_factory or self._default_factory
        self._lock = threading.Lock()

    @staticmethod
    def _default_factory():
        from jarvis_v2.agent.runtime import JarvisRuntime

        return JarvisRuntime()

    def _get_runtime(self):
        with self._lock:
            if self._runtime is None:
                self._runtime = self._runtime_factory()
            return self._runtime

    def submit(self, text: str) -> HudReply:
        text = (text or "").strip()
        if not text:
            return HudReply(response="", error="empty_command")
        try:
            runtime = self._get_runtime()
        except Exception:
            return HudReply(
                response=(
                    f"Jarvis could not start. Run `{V3_BOOTSTRAP_CHECK_COMMAND}`, repair the reported "
                    "storage configuration, then retry."
                ),
                error="runtime_initialization_failed",
            )
        try:
            result = runtime.handle(text)
        except Exception:
            return HudReply(response="Jarvis could not confirm that command's outcome. Do not retry automatically; run `execution health report` and verify any approval chain first.", error="runtime_command_outcome_unknown")
        approval_ids = []
        try:
            if any(item.metadata.get("requires_confirmation") for item in result.tool_results):
                approval_ids = _approval_ids_from_result(result)
        except Exception:
            approval_ids = []
        return HudReply(response=str(result.response or ""), approval_ids=approval_ids)

    def approve(self, approval_id: int) -> HudReply:
        return self.submit(f"approve approval {int(approval_id)}")

    def dismiss(self, approval_id: int) -> HudReply:
        return self.submit(f"dismiss approval {int(approval_id)}")


class HudRecorder:
    """Push-to-talk for the HUD: ffmpeg mic capture → local whisper transcript.

    UI-agnostic and headlessly testable. Reuses talk.py's mic selection, the
    hard `-t` duration cap, and the whisper transcriber. `start()` begins an
    async recording; `stop_and_transcribe()` finalizes the clip (ffmpeg stops
    cleanly on a 'q' keypress) and returns (transcript, error_msg) tuple
    ('', '' on success; transcript='', error_msg set on failure).
    """

    def __init__(self) -> None:
        self._proc: subprocess.Popen | None = None
        self._path: Path | None = None
        self._start_time: float | None = None

    @property
    def recording(self) -> bool:
        return self._proc is not None

    def start(self) -> bool:
        if self._proc is not None:
            return True
        from jarvis_v2.scripts.talk import _default_mic_index, _max_record_seconds
        import time

        try:
            fd, raw_path = tempfile.mkstemp(suffix=".wav")
            os.close(fd)
            dest = Path(raw_path)
            cmd = [
                "ffmpeg", "-loglevel", "error", "-y",
                "-f", "avfoundation", "-i", f":{_default_mic_index()}",
                "-ac", "1", "-ar", "16000",
                "-t", str(_max_record_seconds()),
                str(dest),
            ]
            self._proc = subprocess.Popen(
                cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
        except Exception:
            self._proc = None
            return False
        self._path = dest
        self._start_time = time.time()
        return True

    def stop_and_transcribe(self) -> tuple[str, str]:
        """Stop recording and transcribe. Returns (transcript, error_msg)."""
        import time

        proc, path, start_time = self._proc, self._path, self._start_time
        self._proc, self._path, self._start_time = None, None, None
        if proc is None:
            return ("", "recording not started")

        try:
            proc.communicate(input=b"q", timeout=15)
        except Exception as exc:
            try:
                proc.kill()
                proc.wait(timeout=5)
            except Exception:
                pass
            return ("", f"ffmpeg error: {type(exc).__name__}")

        try:
            if not (path and path.exists()):
                return ("", "no audio file recorded")
            size = path.stat().st_size
            if size == 0:
                return ("", "no audio detected (silent recording)")
            if size < 1000:  # suspiciously small
                return ("", f"audio too short ({size} bytes)")

            from jarvis_v2.scripts.talk import _transcribe

            elapsed = time.time() - start_time if start_time else 0
            text = ""
            try:
                text = (_transcribe(path) or "").strip()
            except subprocess.TimeoutExpired:
                return ("", "transcription timeout (>5min)")
            except FileNotFoundError:
                return ("", "whisper CLI not found (install whisper)")
            except Exception as exc:
                return ("", f"transcription failed: {type(exc).__name__}")

            if not text:
                return ("", "transcription returned empty (inaudible or noise only)")
            return (text, "")
        finally:
            try:
                if path:
                    path.unlink()
            except Exception:
                pass


def stop_speech() -> None:
    """Silence any in-flight `say` speech immediately (mute mid-reply)."""
    try:
        subprocess.run(["pkill", "-x", "say"], capture_output=True, timeout=5)
    except Exception:
        pass


def _speak(text: str) -> None:
    """Speak a reply aloud, replacing any speech already playing."""
    stop_speech()
    try:
        from jarvis_v2.tools.voice import speak_text

        speak_text(text, wait=False)
    except Exception:
        pass


def _speak_default_on() -> bool:
    return os.getenv("JARVIS_HUD_SPEAK", "1") not in {"0", "false", "no", "off"}


# --------------------------------------------------------------------------
# Native AppKit HUD (primary): WhisperFlow and macOS dictation can insert into
# NSTextField because it is a real accessibility text element.
# --------------------------------------------------------------------------


def _run_hud_appkit() -> None:  # pragma: no cover - interactive UI
    import objc
    from AppKit import (
        NSApp,
        NSApplication,
        NSApplicationActivationPolicyAccessory,
        NSBackingStoreBuffered,
        NSButton,
        NSButtonTypeMomentaryPushIn,
        NSButtonTypeSwitch,
        NSColor,
        NSFloatingWindowLevel,
        NSFont,
        NSFontAttributeName,
        NSForegroundColorAttributeName,
        NSMakeRect,
        NSScreen,
        NSScrollView,
        NSTextField,
        NSTextView,
        NSTimer,
        NSWindow,
        NSWindowStyleMaskClosable,
        NSWindowStyleMaskMiniaturizable,
        NSWindowStyleMaskResizable,
        NSWindowStyleMaskTitled,
    )
    from Foundation import NSAttributedString, NSObject

    BG = NSColor.colorWithCalibratedRed_green_blue_alpha_(0.063, 0.078, 0.094, 1.0)
    PANEL = NSColor.colorWithCalibratedRed_green_blue_alpha_(0.102, 0.125, 0.153, 1.0)
    FG = NSColor.colorWithCalibratedRed_green_blue_alpha_(0.91, 0.93, 0.95, 1.0)
    ACCENT = NSColor.colorWithCalibratedRed_green_blue_alpha_(0.29, 0.64, 1.0, 1.0)
    DIM = NSColor.colorWithCalibratedRed_green_blue_alpha_(0.54, 0.59, 0.65, 1.0)
    MONO = NSFont.fontWithName_size_("Menlo", 12.0) or NSFont.systemFontOfSize_(12.0)

    core = HudCore()
    inbox: "queue.Queue[tuple[str, str]]" = queue.Queue()
    outbox: "queue.Queue[HudReply | str]" = queue.Queue()

    def worker() -> None:
        while True:
            kind, payload = inbox.get()
            if kind == "quit":
                return
            outbox.put("__busy__")
            if kind == "approve":
                reply = core.approve(int(payload))
            elif kind == "dismiss":
                reply = core.dismiss(int(payload))
            else:
                reply = core.submit(payload)
            outbox.put(reply)

    threading.Thread(target=worker, daemon=True).start()

    class HudController(NSObject):
        def init(self):
            self = objc.super(HudController, self).init()
            if self is None:
                return None
            self.speak_enabled = _speak_default_on()
            self.approval_buttons = []
            self.recorder = HudRecorder()
            return self

        # -- UI construction ------------------------------------------------
        def buildWindow(self):
            width, height = 470.0, 340.0
            screen = NSScreen.mainScreen().visibleFrame()
            x = screen.origin.x + screen.size.width - width - 24
            y = screen.origin.y + screen.size.height - height - 16
            style = (
                NSWindowStyleMaskTitled
                | NSWindowStyleMaskClosable
                | NSWindowStyleMaskMiniaturizable
                | NSWindowStyleMaskResizable
            )
            self.window = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
                NSMakeRect(x, y, width, height), style, NSBackingStoreBuffered, False
            )
            self.window.setTitle_("Jarvis")
            self.window.setLevel_(NSFloatingWindowLevel)
            self.window.setBackgroundColor_(BG)
            self.window.setDelegate_(self)
            content = self.window.contentView()

            # Conversation log
            self.scroll = NSScrollView.alloc().initWithFrame_(NSMakeRect(10, 96, width - 20, height - 106))
            self.scroll.setHasVerticalScroller_(True)
            self.scroll.setAutoresizingMask_(18)  # width + height sizable
            self.log = NSTextView.alloc().initWithFrame_(NSMakeRect(0, 0, width - 20, height - 106))
            self.log.setEditable_(False)
            self.log.setBackgroundColor_(PANEL)
            self.log.setFont_(MONO)
            self.log.setTextContainerInset_((8, 8))
            self.scroll.setDocumentView_(self.log)
            content.addSubview_(self.scroll)

            # Approvals row (populated on demand)
            self.approvals_y = 66.0

            # 🎤 push-to-talk: click to record, click again to stop + transcribe
            self.mic_button = NSButton.alloc().initWithFrame_(NSMakeRect(10, 33, 40, 28))
            self.mic_button.setButtonType_(NSButtonTypeMomentaryPushIn)
            self.mic_button.setTitle_("🎤")
            self.mic_button.setToolTip_("Speak to Jarvis (click, talk, click again)")
            self.mic_button.setTarget_(self)
            self.mic_button.setAction_("onMic:")
            content.addSubview_(self.mic_button)

            # Input field — a NATIVE text field so WhisperFlow can dictate into it
            self.entry = NSTextField.alloc().initWithFrame_(NSMakeRect(54, 34, width - 160, 26))
            self.entry.setFont_(NSFont.fontWithName_size_("Menlo", 13.0) or NSFont.systemFontOfSize_(13.0))
            self.entry.setBackgroundColor_(PANEL)
            self.entry.setTextColor_(FG)
            self.entry.setPlaceholderString_("Message Jarvis…")
            self.entry.setTarget_(self)
            self.entry.setAction_("onEnter:")
            self.entry.setAutoresizingMask_(2)  # width sizable
            content.addSubview_(self.entry)

            # 🔊 toggle (speak future replies; toggling OFF also silences now)
            self.speak_button = NSButton.alloc().initWithFrame_(NSMakeRect(width - 100, 34, 44, 26))
            self.speak_button.setButtonType_(NSButtonTypeSwitch)
            self.speak_button.setTitle_("🔊")
            self.speak_button.setState_(1 if self.speak_enabled else 0)
            self.speak_button.setTarget_(self)
            self.speak_button.setAction_("onSpeakToggle:")
            self.speak_button.setAutoresizingMask_(1)  # min-x margin sizable
            content.addSubview_(self.speak_button)

            # ⏹ stop speaking NOW (mute a long reply mid-sentence)
            self.stop_button = NSButton.alloc().initWithFrame_(NSMakeRect(width - 52, 33, 42, 28))
            self.stop_button.setButtonType_(NSButtonTypeMomentaryPushIn)
            self.stop_button.setTitle_("⏹")
            self.stop_button.setToolTip_("Stop speaking")
            self.stop_button.setTarget_(self)
            self.stop_button.setAction_("onStopSpeech:")
            self.stop_button.setAutoresizingMask_(1)
            content.addSubview_(self.stop_button)

            # Status line
            self.status = NSTextField.labelWithString_("Ready — dictate with WhisperFlow, press Enter")
            self.status.setFrame_(NSMakeRect(12, 8, width - 24, 18))
            self.status.setFont_(NSFont.fontWithName_size_("Menlo", 10.0) or NSFont.systemFontOfSize_(10.0))
            self.status.setTextColor_(DIM)
            self.status.setAutoresizingMask_(2)
            content.addSubview_(self.status)

            self.appendLine_color_("Jarvis HUD online — same safety gates as Telegram.", DIM)
            self.window.makeKeyAndOrderFront_(None)
            NSApp.activateIgnoringOtherApps_(True)
            self.window.makeFirstResponder_(self.entry)

        # -- helpers ---------------------------------------------------------
        def appendLine_color_(self, text, color):
            attrs = {NSForegroundColorAttributeName: color, NSFontAttributeName: MONO}
            line = NSAttributedString.alloc().initWithString_attributes_(text + "\n", attrs)
            self.log.textStorage().appendAttributedString_(line)
            self.log.scrollRangeToVisible_((self.log.string().length(), 0))

        def windowWillClose_(self, notification):
            # Accessory apps have no Dock icon — quit with the window so the
            # LaunchAgent can relaunch it cleanly.
            stop_speech()
            NSApp.terminate_(None)

        def clearApprovalButtons(self):
            for view in self.approval_buttons:
                view.removeFromSuperview()
            self.approval_buttons = []

        def showApprovals_(self, ids):
            self.clearApprovalButtons()
            content = self.window.contentView()
            x = 10.0
            for approval_id in ids:
                label = NSTextField.labelWithString_(f"Approval #{approval_id}")
                label.setFrame_(NSMakeRect(x, self.approvals_y, 110, 22))
                label.setTextColor_(ACCENT)
                label.setFont_(MONO)
                content.addSubview_(label)
                approve = NSButton.alloc().initWithFrame_(NSMakeRect(x + 112, self.approvals_y - 2, 84, 26))
                approve.setTitle_("Approve")
                approve.setTag_(int(approval_id))
                approve.setTarget_(self)
                approve.setAction_("onApprove:")
                content.addSubview_(approve)
                dismiss = NSButton.alloc().initWithFrame_(NSMakeRect(x + 198, self.approvals_y - 2, 84, 26))
                dismiss.setTitle_("Dismiss")
                dismiss.setTag_(int(approval_id))
                dismiss.setTarget_(self)
                dismiss.setAction_("onDismiss:")
                content.addSubview_(dismiss)
                self.approval_buttons.extend([label, approve, dismiss])
                break  # one approval row at a time keeps the layout sane

        # -- actions -----------------------------------------------------------
        def onEnter_(self, sender):
            text = str(self.entry.stringValue()).strip()
            if not text:
                return
            self.entry.setStringValue_("")
            self.appendLine_color_(f"You: {text}", ACCENT)
            inbox.put(("command", text))

        def onMic_(self, sender):
            if not self.recorder.recording:
                stop_speech()  # don't record Jarvis's own voice
                if self.recorder.start():
                    sender.setTitle_("⏺")
                    self.status.setStringValue_("● Listening — click ⏺ when you're done")
                else:
                    self.status.setStringValue_("Mic unavailable — check ffmpeg and Microphone permission")
            else:
                sender.setTitle_("🎤")
                self.status.setStringValue_("Transcribing (0s)…")

                def finish() -> None:
                    import time
                    start = time.time()
                    # Update status with elapsed time while waiting
                    for i in range(60):
                        elapsed = int(time.time() - start)
                        self.status.setStringValue_(f"Transcribing ({elapsed}s)…")
                        time.sleep(0.5)
                        if elapsed > 30:
                            break  # give up waiting; just show the result when ready
                    text, error = self.recorder.stop_and_transcribe()
                    outbox.put(("voice", text, error))

                threading.Thread(target=finish, daemon=True).start()

        def onSpeakToggle_(self, sender):
            self.speak_enabled = bool(sender.state())
            if not self.speak_enabled:
                stop_speech()

        def onStopSpeech_(self, sender):
            stop_speech()

        def onApprove_(self, sender):
            self.clearApprovalButtons()
            self.appendLine_color_(f"[approve approval #{sender.tag()}]", DIM)
            inbox.put(("approve", str(sender.tag())))

        def onDismiss_(self, sender):
            self.clearApprovalButtons()
            self.appendLine_color_(f"[dismiss approval #{sender.tag()}]", DIM)
            inbox.put(("dismiss", str(sender.tag())))

        # -- outbox polling ----------------------------------------------------
        def poll_(self, timer):
            try:
                while True:
                    item = outbox.get_nowait()
                    if item == "__busy__":
                        self.status.setStringValue_("Jarvis is thinking…")
                        continue
                    if isinstance(item, tuple) and len(item) >= 2 and item[0] == "voice":
                        transcript = (item[1] or "").strip()
                        error = (item[2] or "").strip() if len(item) > 2 else ""
                        if transcript:
                            self.appendLine_color_(f"You (voice): {transcript}", ACCENT)
                            inbox.put(("command", transcript))
                            self.status.setStringValue_("Ready — dictate with WhisperFlow, press Enter")
                        else:
                            msg = error if error else "Didn't catch that — click 🎤 and try again"
                            self.appendLine_color_(f"[Voice error: {msg}]", DIM)
                            self.status.setStringValue_("Ready — dictate with WhisperFlow, press Enter")
                        continue
                    reply = item
                    self.status.setStringValue_("Ready — dictate with WhisperFlow, press Enter")
                    if reply.response:
                        self.appendLine_color_(f"Jarvis: {reply.response}", FG)
                        if self.speak_enabled:
                            _speak(reply.response)
                    if reply.approval_ids:
                        self.showApprovals_(reply.approval_ids)
            except queue.Empty:
                pass

    app = NSApplication.sharedApplication()
    app.setActivationPolicy_(NSApplicationActivationPolicyAccessory)
    controller = HudController.alloc().init()
    controller.buildWindow()
    NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
        0.12, controller, "poll:", None, True
    )
    app.run()
    inbox.put(("quit", ""))


# --------------------------------------------------------------------------
# Tk fallback (typing-only: Tk fields are invisible to dictation tools)
# --------------------------------------------------------------------------


def _run_hud_tk() -> None:  # pragma: no cover - interactive UI
    import tkinter as tk
    from tkinter import ttk

    BG, PANEL, FG, ACCENT, DIM = "#101418", "#1a2027", "#e8edf2", "#4aa3ff", "#8a97a5"

    core = HudCore()
    inbox: "queue.Queue[tuple[str, str]]" = queue.Queue()
    outbox: "queue.Queue[HudReply | str]" = queue.Queue()

    def worker() -> None:
        while True:
            kind, payload = inbox.get()
            if kind == "quit":
                return
            outbox.put("__busy__")
            if kind == "approve":
                reply = core.approve(int(payload))
            elif kind == "dismiss":
                reply = core.dismiss(int(payload))
            else:
                reply = core.submit(payload)
            outbox.put(reply)

    threading.Thread(target=worker, daemon=True).start()

    root = tk.Tk()
    root.title("Jarvis")
    root.configure(bg=BG)
    root.attributes("-topmost", True)
    width, height = 460, 320
    screen_w = root.winfo_screenwidth()
    root.geometry(f"{width}x{height}+{screen_w - width - 24}+40")

    log = tk.Text(
        root, bg=PANEL, fg=FG, insertbackground=FG, relief="flat", wrap="word",
        font=("Menlo", 12), padx=10, pady=8, state="disabled", height=10,
    )
    log.pack(fill="both", expand=True, padx=10, pady=(10, 6))
    log.tag_configure("you", foreground=ACCENT)
    log.tag_configure("jarvis", foreground=FG)
    log.tag_configure("dim", foreground=DIM)

    approvals_frame = tk.Frame(root, bg=BG)
    approvals_frame.pack(fill="x", padx=10)

    bottom = tk.Frame(root, bg=BG)
    bottom.pack(fill="x", padx=10, pady=(0, 10))
    recorder = HudRecorder()

    def on_mic() -> None:
        if not recorder.recording:
            stop_speech()  # don't record Jarvis's own voice
            if recorder.start():
                mic_btn.configure(text="⏺")
                status.configure(text="● Listening — click ⏺ when you're done")
            else:
                status.configure(text="Mic unavailable — check ffmpeg and Microphone permission")
        else:
            mic_btn.configure(text="🎤")
            status.configure(text="Transcribing (0s)…")

            def finish() -> None:
                import time
                start = time.time()
                for i in range(60):
                    elapsed = int(time.time() - start)
                    status.configure(text=f"Transcribing ({elapsed}s)…")
                    root.update()
                    time.sleep(0.5)
                    if elapsed > 30:
                        break
                text, error = recorder.stop_and_transcribe()
                outbox.put(("voice", text, error))

            threading.Thread(target=finish, daemon=True).start()

    mic_btn = tk.Button(bottom, text="🎤", command=on_mic, highlightbackground=BG)
    mic_btn.pack(side="left", padx=(0, 6))
    entry = tk.Entry(bottom, bg=PANEL, fg=FG, insertbackground=FG, relief="flat", font=("Menlo", 13))
    entry.pack(side="left", fill="x", expand=True, ipady=6)
    speak_var = tk.BooleanVar(value=_speak_default_on())

    def on_speak_toggle() -> None:
        if not speak_var.get():
            stop_speech()

    speak_box = ttk.Checkbutton(bottom, text="🔊", variable=speak_var, command=on_speak_toggle)
    speak_box.pack(side="right", padx=(6, 0))
    stop_btn = tk.Button(bottom, text="⏹", command=stop_speech, highlightbackground=BG)
    stop_btn.pack(side="right", padx=(6, 0))
    status = tk.Label(root, text="Ready — type a command, press Enter (dictation needs the AppKit HUD)", bg=BG, fg=DIM, font=("Menlo", 10))
    status.pack(fill="x", padx=12, pady=(0, 6))

    def append(text: str, tag: str) -> None:
        log.configure(state="normal")
        log.insert("end", text + "\n", tag)
        log.configure(state="disabled")
        log.see("end")

    def clear_approvals() -> None:
        for child in approvals_frame.winfo_children():
            child.destroy()

    def show_approvals(ids: list[int]) -> None:
        clear_approvals()
        for approval_id in ids:
            row = tk.Frame(approvals_frame, bg=BG)
            row.pack(fill="x", pady=2)
            tk.Label(row, text=f"Approval #{approval_id} pending", bg=BG, fg=ACCENT, font=("Menlo", 11)).pack(side="left")
            tk.Button(row, text="Approve", command=lambda a=approval_id: send_action("approve", a), highlightbackground=BG).pack(side="right", padx=2)
            tk.Button(row, text="Dismiss", command=lambda a=approval_id: send_action("dismiss", a), highlightbackground=BG).pack(side="right", padx=2)

    def send_action(kind: str, approval_id: int) -> None:
        clear_approvals()
        append(f"[{kind} approval #{approval_id}]", "dim")
        inbox.put((kind, str(approval_id)))

    def on_enter(_event=None) -> None:
        text = entry.get().strip()
        if not text:
            return
        entry.delete(0, "end")
        append(f"You: {text}", "you")
        inbox.put(("command", text))

    def poll_outbox() -> None:
        try:
            while True:
                item = outbox.get_nowait()
                if item == "__busy__":
                    status.configure(text="Jarvis is thinking…")
                    continue
                if isinstance(item, tuple) and len(item) >= 2 and item[0] == "voice":
                    transcript = (item[1] or "").strip()
                    error = (item[2] or "").strip() if len(item) > 2 else ""
                    if transcript:
                        append(f"You (voice): {transcript}", "you")
                        inbox.put(("command", transcript))
                        status.configure(text="Ready — type a command, press Enter")
                    else:
                        msg = error if error else "Didn't catch that — click 🎤 and try again"
                        append(f"[Voice error: {msg}]", "dim")
                        status.configure(text="Ready — type a command, press Enter")
                    continue
                reply = item
                status.configure(text="Ready — type a command, press Enter")
                if reply.response:
                    append(f"Jarvis: {reply.response}", "jarvis")
                    if speak_var.get():
                        _speak(reply.response)
                if reply.approval_ids:
                    show_approvals(reply.approval_ids)
        except queue.Empty:
            pass
        root.after(120, poll_outbox)

    entry.bind("<Return>", on_enter)
    entry.focus_set()
    append("Jarvis HUD online (Tk fallback — typing only).", "dim")
    root.after(120, poll_outbox)
    root.mainloop()
    inbox.put(("quit", ""))


def run_hud() -> None:  # pragma: no cover - interactive UI
    # .env must be in os.environ BEFORE the warmup/transcriber checks below —
    # the runtime loads it lazily on the first command, which is too late for
    # startup warmup (and the mic would fall back to the slow whisper CLI).
    try:
        from jarvis_v2.env import load_env

        load_env()
    except Exception:
        pass
    # Pre-load the whisper model in the background (opt-in via
    # JARVIS_VOICE_WARMUP) so the FIRST mic transcription is ~0.5s instead of
    # paying the model load. No mic access — it transcribes a silent wav.
    try:
        from jarvis_v2.scripts.talk import start_voice_warmup_if_enabled

        start_voice_warmup_if_enabled(source="hud")
    except Exception:
        pass
    try:
        import AppKit  # noqa: F401
        import objc  # noqa: F401
    except Exception:
        return _run_hud_tk()
    return _run_hud_appkit()


def main() -> None:  # pragma: no cover - interactive UI
    run_hud()


if __name__ == "__main__":
    main()
