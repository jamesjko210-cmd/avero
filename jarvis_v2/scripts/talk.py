"""Talk to Jarvis directly from the Mac — local push-to-talk, no Telegram.

Press Enter to start recording from the mic, Enter again to stop. Jarvis
transcribes the clip with the local whisper CLI and runs the transcript through
the SAME approval-gated runtime as a typed command, then prints (and optionally
speaks) the reply. Loops until Ctrl-C / EOF.

    python3 -m jarvis_v2.scripts.talk            # type/print loop
    python3 -m jarvis_v2.scripts.talk --speak    # also speak replies aloud
    python3 -m jarvis_v2.scripts.talk --language ko  # force Korean ASR for this local session
    python3 -m jarvis_v2.scripts.talk --list-mics
    JARVIS_VOICE_MIC=1 python3 -m jarvis_v2.scripts.talk   # force a mic index

Requirements: `ffmpeg` (mic capture) plus one configured local transcriber. Run
``python3 -m jarvis_v2.scripts.talk --check`` before the first recording. The
check does not open the microphone or inspect private data.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import re
import shutil
import subprocess
import tempfile
import threading
import wave
from pathlib import Path
from typing import Callable


_VOICE_WARMUP_STARTED = False
_VOICE_WARMUP_LOCK = threading.Lock()
_VOICE_WARMUP_TRUE_VALUES = {"1", "true", "yes", "on"}
_VOICE_WARMUP_SECONDS = 0.3
_VOICE_WARMUP_SAMPLE_RATE = 16000
_VOICE_READINESS_ENV_KEYS = frozenset(
    {
        "JARVIS_VOICE_FASTER_WHISPER_MODEL_PATH",
        "JARVIS_VOICE_MAX_SECONDS",
        "JARVIS_VOICE_MIC",
        "JARVIS_VOICE_WHISPER_CLI",
        "JARVIS_VOICE_WHISPER_CLI_LANGUAGE",
        "JARVIS_VOICE_WHISPER_CLI_MODEL",
        "JARVIS_VOICE_WHISPER_MODEL_PATH",
    }
)


def _voice_readiness_environment(
    environ: dict[str, str] | None = None,
) -> tuple[dict[str, str], str]:
    """Merge only voice settings from a selected V3 env, returning a safe error."""

    env = dict(os.environ if environ is None else environ)
    configured_path = str(env.get("JARVIS_V3_ENV") or "").strip()
    if not configured_path:
        return env, ""
    try:
        from jarvis_v2.env import read_env_values

        selected = read_env_values(Path(configured_path))
    except Exception:
        return env, "the selected V3 environment could not be read safely"
    for key in _VOICE_READINESS_ENV_KEYS:
        if key not in env and key in selected:
            env[key] = selected[key]
    return env, ""


def _voice_readiness_snapshot(
    *,
    environ: dict[str, str] | None = None,
    command_finder: Callable[[str], str | None] = shutil.which,
    module_finder: Callable[[str], object | None] = importlib.util.find_spec,
    environment_issue: str = "",
) -> dict[str, object]:
    """Inspect push-to-talk prerequisites without opening audio devices.

    The result intentionally contains only fixed labels and booleans. Configured
    executable/model paths and probe exceptions never enter the report.
    """

    env = dict(os.environ if environ is None else environ)
    issues: list[str] = []
    recovery: list[str] = []

    try:
        ffmpeg_ready = bool(command_finder("ffmpeg"))
    except Exception:
        ffmpeg_ready = False
    try:
        speech_ready = bool(command_finder("say"))
    except Exception:
        speech_ready = False

    whisper_model_raw = str(env.get("JARVIS_VOICE_WHISPER_MODEL_PATH") or "").strip()
    faster_model_raw = str(env.get("JARVIS_VOICE_FASTER_WHISPER_MODEL_PATH") or "").strip()
    cli_override_raw = str(env.get("JARVIS_VOICE_WHISPER_CLI") or "").strip()
    cli_model = str(env.get("JARVIS_VOICE_WHISPER_CLI_MODEL") or "base").strip().lower() or "base"
    cli_language = str(env.get("JARVIS_VOICE_WHISPER_CLI_LANGUAGE") or "").strip().lower()

    transcriber_ready = False
    transcriber_source = "not configured"
    configuration_valid = not environment_issue
    if environment_issue:
        issues.append(environment_issue)

    if whisper_model_raw and faster_model_raw:
        configuration_valid = False
        issues.append("more than one local model source is configured")
    elif whisper_model_raw:
        try:
            model_ready = Path(whisper_model_raw).expanduser().is_file()
            package_ready = module_finder("whisper") is not None
        except Exception:
            model_ready = package_ready = False
        transcriber_ready = bool(model_ready and package_ready)
        transcriber_source = "local Whisper model"
        if not model_ready:
            configuration_valid = False
            issues.append("the configured Whisper model file is unavailable")
        if not package_ready:
            configuration_valid = False
            issues.append("the Whisper Python package is unavailable")
    elif faster_model_raw:
        try:
            model_ready = Path(faster_model_raw).expanduser().is_dir()
            package_ready = module_finder("faster_whisper") is not None
        except Exception:
            model_ready = package_ready = False
        transcriber_ready = bool(model_ready and package_ready)
        transcriber_source = "local faster-whisper model"
        if not model_ready:
            configuration_valid = False
            issues.append("the configured faster-whisper model directory is unavailable")
        if not package_ready:
            configuration_valid = False
            issues.append("the faster-whisper Python package is unavailable")
    else:
        try:
            if cli_override_raw:
                cli_path = Path(cli_override_raw).expanduser()
                cli_ready = cli_path.is_file() and os.access(cli_path, os.X_OK)
            else:
                cli_ready = bool(command_finder("whisper"))
        except Exception:
            cli_ready = False
        transcriber_ready = cli_ready
        transcriber_source = "Whisper command" if cli_ready else "not configured"
        if cli_override_raw and not cli_ready:
            configuration_valid = False
            issues.append("the configured Whisper command is unavailable or not executable")

    try:
        from jarvis_v2.tools import voice

        if voice._AUDIO_FILE_TRANSCRIBER is not None and not whisper_model_raw and not faster_model_raw:
            transcriber_ready = True
            transcriber_source = "in-process transcriber"
    except Exception:
        pass

    try:
        from jarvis_v2.tools.voice import SAFE_WHISPER_CLI_MODELS

        safe_cli_models = SAFE_WHISPER_CLI_MODELS
    except Exception:
        safe_cli_models = {"tiny", "base", "small", "medium", "large", "turbo"}
    if cli_model not in safe_cli_models:
        configuration_valid = False
        issues.append("the Whisper command model name is not supported")
    if cli_language and not re.fullmatch(r"[a-z]{2,3}", cli_language):
        configuration_valid = False
        issues.append("the Whisper command language must be a two- or three-letter code")

    raw_seconds = str(env.get("JARVIS_VOICE_MAX_SECONDS") or "120").strip()
    try:
        seconds = int(raw_seconds)
        duration_valid = 5 <= seconds <= 600
    except (TypeError, ValueError):
        duration_valid = False
    if not duration_valid:
        configuration_valid = False
        issues.append("the recording limit must be a whole number from 5 to 600 seconds")

    raw_mic = str(env.get("JARVIS_VOICE_MIC") or "").strip()
    if raw_mic and (not raw_mic.isdigit() or int(raw_mic) < 0):
        configuration_valid = False
        issues.append("the microphone selector must be a non-negative device number")

    if not ffmpeg_ready:
        issues.insert(0, "ffmpeg is unavailable")
        recovery.append("Install ffmpeg, then run this readiness check again.")
    if not transcriber_ready:
        if not any("Whisper" in issue or "whisper" in issue for issue in issues):
            issues.append("no local transcriber is available")
        recovery.append("Install the Whisper command or configure one supported local model, then check again.")
    if not configuration_valid:
        recovery.append("Correct the reported voice setting in the private V3 environment, then check again.")
    recovery.append(
        "When ready, start push-to-talk yourself; macOS may then ask for Microphone access for the terminal application."
    )

    # Keep recovery bounded and deterministic even if multiple settings are bad.
    return {
        "ready": bool(ffmpeg_ready and transcriber_ready and configuration_valid),
        "ffmpeg_ready": ffmpeg_ready,
        "transcriber_ready": transcriber_ready,
        "transcriber_source": transcriber_source,
        "speech_ready": speech_ready,
        "configuration_valid": configuration_valid,
        "issues": tuple(issues[:6]),
        "recovery": tuple(dict.fromkeys(recovery))[:4],
        "microphone_opened": False,
        "audio_recorded": False,
        "private_data_read": False,
    }


def _format_voice_readiness(snapshot: dict[str, object]) -> str:
    """Render a bounded, path-free first-run readiness report."""

    yes_no = lambda value: "ready" if value else "missing"
    lines = [
        "Jarvis V3 voice readiness (no microphone opened):",
        f"- ffmpeg capture command: {yes_no(snapshot.get('ffmpeg_ready'))}",
        f"- local transcriber: {yes_no(snapshot.get('transcriber_ready'))}",
        f"- selected transcriber: {snapshot.get('transcriber_source') or 'not configured'}",
        f"- voice configuration: {'valid' if snapshot.get('configuration_valid') else 'needs attention'}",
        f"- spoken reply command: {yes_no(snapshot.get('speech_ready'))}",
        "- microphone permission: not requested or probed by this check",
    ]
    issues = tuple(snapshot.get("issues") or ())[:6]
    if issues:
        lines.extend(["", "Needs attention:"])
        lines.extend(f"- {issue}" for issue in issues)
    lines.extend(["", "Next steps:"])
    lines.extend(f"- {step}" for step in tuple(snapshot.get("recovery") or ())[:4])
    lines.extend(
        [
            "",
            "Safety boundary: this check does not list microphones, open the microphone, record or transcribe audio,",
            "start Jarvis runtime, read private data, or write files.",
            f"Overall: {'READY' if snapshot.get('ready') else 'NOT READY'}",
        ]
    )
    return "\n".join(lines)


def _list_audio_devices() -> list[tuple[int, str]]:
    """Parse `ffmpeg -f avfoundation -list_devices` output into (index, name) pairs."""
    proc = subprocess.run(
        ["ffmpeg", "-f", "avfoundation", "-list_devices", "true", "-i", ""],
        capture_output=True,
        text=True,
        timeout=20,
    )
    devices: list[tuple[int, str]] = []
    in_audio = False
    for line in proc.stderr.splitlines():
        if "AVFoundation audio devices" in line:
            in_audio = True
            continue
        if in_audio:
            m = re.search(r"\[(\d+)\]\s+(.*)$", line)
            if m:
                devices.append((int(m.group(1)), m.group(2).strip()))
            elif "AVFoundation" not in line:
                break
    return devices


def _default_mic_index() -> str:
    import os

    override = str(os.environ.get("JARVIS_VOICE_MIC") or "").strip()
    if override:
        return override
    devices = _list_audio_devices()
    # Prefer the built-in mic; avoid Continuity/iPhone mics that may not be present.
    for idx, name in devices:
        low = name.lower()
        if "macbook" in low or "built-in" in low:
            return str(idx)
    for idx, name in devices:
        if "iphone" not in name.lower():
            return str(idx)
    return str(devices[0][0]) if devices else "0"


def _max_record_seconds() -> int:
    try:
        value = int(str(os.environ.get("JARVIS_VOICE_MAX_SECONDS") or "120").strip())
    except (TypeError, ValueError):
        return 120
    return value if 5 <= value <= 600 else 120


def _voice_warmup_enabled() -> bool:
    return str(os.environ.get("JARVIS_VOICE_WARMUP") or "").strip().lower() in _VOICE_WARMUP_TRUE_VALUES


def _write_silent_warmup_wav(path: Path) -> None:
    frame_count = max(1, int(_VOICE_WARMUP_SAMPLE_RATE * _VOICE_WARMUP_SECONDS))
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(_VOICE_WARMUP_SAMPLE_RATE)
        wav.writeframes(b"\x00\x00" * frame_count)


def _run_voice_warmup(*, transcriber: Callable[[Path], str] | None = None) -> None:
    if transcriber is None:
        from jarvis_v2.tools import voice

        transcriber = voice._audio_file_transcriber()
    if transcriber is None:
        return
    try:
        with tempfile.TemporaryDirectory(prefix="jarvis-voice-warmup-") as tmp:
            clip = Path(tmp) / "silence-0.3s.wav"
            _write_silent_warmup_wav(clip)
            transcriber(clip)
    except Exception:
        return


def start_voice_warmup_if_enabled(
    *,
    source: str = "ui",
    thread_factory: Callable[..., threading.Thread] = threading.Thread,
) -> bool:
    """Start one opt-in background audio-file warmup without touching the mic."""
    global _VOICE_WARMUP_STARTED
    if not _voice_warmup_enabled():
        return False
    with _VOICE_WARMUP_LOCK:
        if _VOICE_WARMUP_STARTED:
            return False
        _VOICE_WARMUP_STARTED = True
    thread_name = f"jarvis-voice-warmup-{re.sub(r'[^a-zA-Z0-9_.-]+', '-', source).strip('-') or 'ui'}"
    try:
        worker = thread_factory(target=_run_voice_warmup, name=thread_name, daemon=True)
        worker.start()
    except Exception:
        with _VOICE_WARMUP_LOCK:
            _VOICE_WARMUP_STARTED = False
        return False
    return True


def _record_clip(mic_index: str, dest: Path) -> bool:
    """Record from the mic until the user presses Enter. Returns True on success.

    A hard `-t` cap stops ffmpeg on its own so a forgotten (or permission-stuck)
    recording can't run unbounded or hand whisper an oversized clip.
    """
    cmd = [
        "ffmpeg", "-loglevel", "error", "-y",
        "-f", "avfoundation", "-i", f":{mic_index}",
        "-ac", "1", "-ar", "16000",
        "-t", str(_max_record_seconds()),
        str(dest),
    ]
    try:
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    except Exception as exc:
        print(f"  (couldn't start ffmpeg: {exc})")
        return False
    print("  ● recording — press Enter to stop…", end="", flush=True)
    try:
        input()
    except (EOFError, KeyboardInterrupt):
        pass
    try:
        # 'q' tells ffmpeg to stop cleanly and finalize the file.
        proc.communicate(input=b"q", timeout=15)
    except Exception:
        proc.kill()
        proc.wait()
    return dest.exists() and dest.stat().st_size > 0


def _transcribe(path: Path, *, language: str = "auto") -> str:
    from jarvis_v2.tools import voice

    transcriber = voice._audio_file_transcriber(language=None if language == "auto" else language)
    if transcriber is None:
        print("  (no transcriber configured — install the `whisper` CLI)")
        return ""
    try:
        return (transcriber(path) or "").strip()
    except Exception as exc:
        print(f"  (transcription failed: {type(exc).__name__})")
        return ""


def run_turn(runtime, transcript: str, *, speak: bool) -> None:
    """Route one transcript through the runtime and surface the reply.

    Voice never auto-approves a side effect: if the spoken command queues an
    approval, we say so and point to the deliberate approve path (Telegram inline
    buttons or the dashboard) rather than acting on a transcribed 'yes'.
    """
    result = runtime.handle(transcript)
    reply = getattr(result, "response", "") or ""
    print(f"\nJarvis: {reply}")

    spoken = reply
    try:
        from jarvis_v2.automations import telegram_control as tg

        approval_id = tg._approval_id_from_result(result)
    except Exception:
        approval_id = None
    if approval_id is not None:
        note = (
            f"⏳ That needs your approval (#{approval_id}). I won't act on it from voice — "
            "approve it in Telegram (✅/❌) or the dashboard."
        )
        print(note)
        spoken = "That needs your approval. Approve it in Telegram or the dashboard."
    print()

    if speak and spoken:
        try:
            from jarvis_v2.tools import voice

            speech_result = voice.speak_text(spoken)
            if not speech_result.ok:
                print(f"Speech output failed: {speech_result.output}")
        except Exception:
            print("Speech output failed. Check macOS speech settings, then try again.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Talk to Jarvis locally via push-to-talk.")
    speech = parser.add_mutually_exclusive_group()
    speech.add_argument("--speak", dest="speak", action="store_true", help="Speak Jarvis's replies aloud.")
    speech.add_argument("--no-speak", dest="speak", action="store_false", help="Print replies without speaking them.")
    parser.set_defaults(speak=False)
    parser.add_argument(
        "--language",
        choices=("auto", "en", "ko"),
        default="auto",
        help="ASR language for this local session; auto preserves multilingual detection.",
    )
    parser.add_argument("--list-mics", action="store_true", help="List available microphones and exit.")
    parser.add_argument(
        "--check",
        "--readiness",
        dest="check",
        action="store_true",
        help="Check voice prerequisites without opening the microphone or starting Jarvis.",
    )
    parser.add_argument("--mic", default="", help="Audio device index to record from (overrides auto-detect).")
    args = parser.parse_args()

    if args.check:
        check_env, environment_issue = _voice_readiness_environment()
        if args.mic:
            check_env["JARVIS_VOICE_MIC"] = args.mic
        snapshot = _voice_readiness_snapshot(environ=check_env, environment_issue=environment_issue)
        print(_format_voice_readiness(snapshot))
        if not snapshot["ready"]:
            raise SystemExit(2)
        return

    if not shutil.which("ffmpeg"):
        raise SystemExit("ffmpeg is required for mic capture. Install it (brew install ffmpeg).")

    if args.list_mics:
        for idx, name in _list_audio_devices():
            print(f"  [{idx}] {name}")
        return

    # The real launch path enforces the same readiness contract as ``--check``.
    # Documentation is helpful, but skipping a documented preflight must never
    # let Jarvis open a microphone when transcription is unavailable or the
    # bounded voice configuration is invalid.
    check_env, environment_issue = _voice_readiness_environment()
    if args.mic:
        check_env["JARVIS_VOICE_MIC"] = args.mic
    snapshot = _voice_readiness_snapshot(environ=check_env, environment_issue=environment_issue)
    if not snapshot["ready"]:
        print(_format_voice_readiness(snapshot))
        raise SystemExit(2)

    if args.mic:
        os.environ["JARVIS_VOICE_MIC"] = args.mic
    mic_index = _default_mic_index()

    from jarvis_v2.agent.runtime import JarvisRuntime
    from jarvis_v2.scripts.startup import is_startup_storage_error, print_startup_failure

    try:
        runtime = JarvisRuntime()
    except Exception as exc:
        if not is_startup_storage_error(exc):
            raise
        print_startup_failure(exc, json_output=False, program="Jarvis talk")
        raise SystemExit(3) from exc

    print(f"Jarvis is listening (mic [{mic_index}]). Press Enter to talk, Ctrl-C to quit.")
    while True:
        try:
            input("\n▶ Press Enter to talk…")
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            return
        with tempfile.TemporaryDirectory(prefix="jarvis-talk-") as tmp:
            clip = Path(tmp) / "clip.wav"
            if not _record_clip(mic_index, clip):
                print("  (no audio captured)")
                continue
            language_label = "" if args.language == "auto" else f" ({args.language})"
            print(f"\r  transcribing{language_label}…            ", flush=True)
            transcript = _transcribe(clip, language=args.language)
        if not transcript:
            print("  (didn't catch that)")
            continue
        print(f"\n🎙 You: {transcript}")
        try:
            run_turn(runtime, transcript, speak=args.speak)
        except Exception as exc:
            print(f"  (runtime error: {type(exc).__name__})")


if __name__ == "__main__":
    main()
