"""Opt-in END-TO-END live self-check for Jarvis's flagship phone-facing features.

Unlike the mocked `smoke_test_*` suite, this actually exercises the real Mac:
macOS Contacts, the live daily-brief composition, and the Telegram owner channel.
It is a DIAGNOSTIC — it is never imported by `smoke_test_all`, never run in CI,
and always exits 0 so it can't break automation. It just prints a ✓/✗ table so
the operator can confirm, in one command, what really works on his machine.

Run (read-only — no Telegram message is sent):
    JARVIS_V3_ENV="$HOME/.jarvis_v3/runtime.env" JARVIS_LIVE_CHECK=1 .venv/bin/python3 -m jarvis_v2.scripts.live_check --contact 가상연락처일

Add a read-only Telegram API connectivity check (calls getMe; sends nothing):
    JARVIS_V3_ENV="$HOME/.jarvis_v3/runtime.env" JARVIS_LIVE_CHECK=1 JARVIS_LIVE_CHECK_TELEGRAM_READ=1 .venv/bin/python3 -m jarvis_v2.scripts.live_check --contact 가상연락처일

Add an actual one-off Telegram test message to the owner:
    JARVIS_V3_ENV="$HOME/.jarvis_v3/runtime.env" JARVIS_LIVE_CHECK=1 JARVIS_LIVE_CHECK_SEND=1 .venv/bin/python3 -m jarvis_v2.scripts.live_check --contact 가상연락처일

The contact name can also come from JARVIS_LIVE_CHECK_CONTACT instead of --contact.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
import json
import math
import os
from pathlib import Path
import plistlib
import pwd
import re
import shlex
import shutil
import stat
import subprocess
import sys
from types import SimpleNamespace
from typing import Callable
import unicodedata

from jarvis_v2.config import load_config
from jarvis_v2.scripts.daemon_gate import (
    V3_DAEMON_ENABLE_ENV,
    V3_SCHEDULER_ENABLE_ENV,
)
from jarvis_v2.v3_commands import (
    V3_CALENDAR_AUTH_FULL_ACCESS_COMMAND,
    V3_CALENDAR_AUTH_READONLY_COMMAND,
    V3_CALENDAR_AUTH_TERMINAL_LOCATION,
    V3_DASHBOARD_COMMAND,
    V3_LIVE_CHECK_COMMAND,
)


LOCAL_PATH_RE = re.compile(r"/(?:Users|private|var/folders|tmp)/[^\n\r;]*", re.IGNORECASE)
MAX_DIAGNOSTIC_CHARS = 120
JOB_FRESHNESS_GRACE_MINUTES = 10
SCHEDULED_JOB_EXPECTED_ENABLED_COUNT = 8
SCHEDULED_JOB_STREAK_DAYS = 7
SCHEDULED_JOB_EXPECTED_INTERVALS = {
    "morning_brief_telegram": 1440,
    "daily_brief": 1440,
    "inbox_ingest": 240,
    "recent_file_digest": 720,
    "goal_nudge": 1440,
    "weekly_review": 10080,
    "state_snapshot": 360,
    "conversation_compaction": 1440,
}
SCHEDULED_JOB_RUN_HISTORY_KEY = "run_history"
SCHEDULED_JOB_RUN_HISTORY_LIMIT = 64
SCHEDULED_JOB_TYPE_RE = re.compile(r"[a-z][a-z0-9_]{0,79}")
SCHEDULED_JOB_RUN_OCCURRENCE_RE = re.compile(
    r"(?:scheduler-occurrence:v1:[0-9a-f]{64}"
    r"|scheduled:scheduler-occurrence:v1:[0-9a-f]{64}"
    r"|scheduled:[0-9]+:[0-9a-f]{32}"
    r"|morning:[0-9a-f]{32}"
    r"|legacy:[0-9a-f]{64})"
)
CHAT_PATH_RECENT_LIMIT = 100
CHAT_PATH_MIN_READY_TURNS = 3
CHAT_LATENCY_METADATA_KEYS = ("latency_ms", "duration_ms", "response_latency_ms", "elapsed_ms")
MIXED_CONVERSATION_RECENT_SESSIONS = 20
MIXED_CONVERSATION_MESSAGE_LIMIT = 80
MIXED_CONVERSATION_MIN_ASSISTANT_TURNS = 10
MIXED_CONVERSATION_MIN_CHAT_LATENCY_SAMPLES = 3
MIXED_CONVERSATION_TARGET_P95_MS = 8000.0
MIXED_CONVERSATION_LATENCY_RECOVERY = (
    " — run `model routing status`; tune JARVIS_CHAT_MAX_REPLY_TOKENS or "
    "JARVIS_CHAT_MAX_HISTORY_MESSAGES only with the operator, then rerun the mixed conversation proof and live_check"
)
MIXED_CONVERSATION_PROOF_RECOVERY = (
    " — proof path: collect one session with chat, research, calendar, tasks, and at least "
    "3 chat latency samples; use `model routing status` for tuning context; rerun the mixed "
    "conversation proof and live_check before claiming done"
)
MIXED_CONVERSATION_REQUIRED_LANES = ("chat", "research", "calendar", "tasks")
MIXED_RESEARCH_TOOLS = {"research", "web_lookup", "web_search", "fetch_page", "wiki_summary"}
MIXED_CALENDAR_TOOLS = {
    "list_calendars",
    "list_events",
    "check_availability",
    "create_event",
    "update_event",
    "delete_event",
}
MIXED_TASK_TOOLS = {
    "add_task",
    "list_tasks",
    "task_overview",
    "task_board",
    "complete_task",
    "task_completion_packet",
    "complete_task_with_evidence",
}
PERSONAL_PROOF_RECENT_LIMIT = 500
PERSONAL_PROOF_CALENDAR_READ_TOOLS = {"list_calendars", "list_events", "check_availability"}
PERSONAL_PROOF_CALENDAR_WRITE_TOOLS = ("create_event", "update_event", "delete_event")
PERSONAL_PROOF_EMAIL_READ_TOOLS = {"read_emails"}
PERSONAL_PROOF_EMAIL_SEARCH_TOOLS = {"search_emails", "read_email_body"}
PERSONAL_PROOF_EMAIL_SEND_TOOLS = {"send_email"}
PERSONAL_PROOF_REMINDER_TOOLS = {"set_reminder"}
PERSONAL_PROOF_RELEVANT_TOOLS = (
    PERSONAL_PROOF_CALENDAR_READ_TOOLS
    | set(PERSONAL_PROOF_CALENDAR_WRITE_TOOLS)
    | PERSONAL_PROOF_EMAIL_READ_TOOLS
    | PERSONAL_PROOF_EMAIL_SEARCH_TOOLS
    | PERSONAL_PROOF_EMAIL_SEND_TOOLS
    | PERSONAL_PROOF_REMINDER_TOOLS
)
PERSONAL_INTEGRATION_RISKS = {
    "list_calendars": "LOCAL_SAFE",
    "list_events": "LOCAL_SAFE",
    "check_availability": "LOCAL_SAFE",
    "create_event": "HIGH_RISK",
    "update_event": "HIGH_RISK",
    "delete_event": "HIGH_RISK",
    "read_emails": "LOCAL_SAFE",
    "search_emails": "PERSONAL_DATA",
    "read_email_body": "PERSONAL_DATA",
    "send_email": "HIGH_RISK",
    "set_reminder": "EXTERNAL_SIDE_EFFECT",
    "list_reminders": "READ_ONLY",
    "cancel_reminders": "LOCAL_SAFE",
    "apple_reminders": "PERSONAL_DATA",
    "create_reminder": "HIGH_RISK",
}
RESEARCH_TOOL_RISKS = {
    "research": "LOCAL_SAFE",
    "web_lookup": "READ_ONLY",
    "web_search": "LOCAL_SAFE",
    "fetch_page": "LOCAL_SAFE",
    "extract_links": "LOCAL_SAFE",
}
RESEARCH_ROUTE_CASES = (
    ("research ada lovelace please", "research", "ada lovelace"),
    ("deep dive on ada lovelace please", "research", "ada lovelace"),
    ("read up on ada lovelace please", "research", "ada lovelace"),
    ("look up the capital of Mongolia please", "web_lookup", "the capital of Mongolia"),
    ("search the web for ada lovelace please", "web_lookup", "ada lovelace"),
    ("web search ada lovelace please", "web_search", "ada lovelace"),
    ("google ada lovelace please", "web_search", "ada lovelace"),
)
PERSONAL_ROUTE_CASES = (
    ("schedule dentist tomorrow at 3pm", "create_event", {"title": "dentist"}, ("start", "end")),
    ("move event evt123 to June 20 at 2pm", "update_event", {"event_id": "evt123"}, ("start", "end")),
    ("rename event evt123 to Lunch with Sam", "update_event", {"event_id": "evt123", "title": "Lunch with Sam"}, ()),
    ("delete event evt123", "delete_event", {"event_id": "evt123"}, ()),
    ("am I free tomorrow afternoon", "check_availability", {"text": "am I free tomorrow afternoon"}, ()),
    ("what is on my calendar tomorrow", "list_events", {"range": "tomorrow"}, ()),
    ("search emails about invoice", "search_emails", {"query": "invoice"}, ()),
    ("read email from Sam about invoice", "read_email_body", {"sender": "Sam", "query": "invoice"}, ()),
    ("check unread email", "read_emails", {"unread": True}, ()),
    ("send email to alex@example.com saying hello", "send_email", {"to": "alex@example.com", "body": "hello"}, ()),
    ("remind me to review approval safety", "create_reminder", {"title": "review approval safety"}, ()),
    ("remind me to drink water in 10 minutes", "set_reminder", {"text": "remind me to drink water in 10 minutes"}, ()),
    ("list reminders", "list_reminders", {}, ()),
    ("cancel reminders", "cancel_reminders", {}, ()),
)
VOICE_TRANSCRIBER_LABELS = {
    "local_whisper_cli": "whisper CLI",
    "local_whisper_model_path": "whisper model (JARVIS_VOICE_WHISPER_MODEL_PATH)",
    "local_faster_whisper_model_path": "faster-whisper model",
    "in_process_transcriber": "in-process transcriber",
}
REPO_ROOT = Path(__file__).resolve().parents[2]
STARTUP_REQUIRED_TEXT = {
    "dashboard launcher": {
        "path": Path("launch_jarvis_v3_dashboard.py"),
        "tokens": ("jarvis_v2.scripts.run_status_server", "main"),
    },
    "Telegram control daemon": {
        "path": Path("jarvis_v2/scripts/run_telegram_control.py"),
        "tokens": ("TelegramCommandBridge", "_start_scheduler_ticker(config)", "run_forever"),
    },
    "scheduler loop": {
        "path": Path("jarvis_v2/scripts/run_scheduler.py"),
        "tokens": ("Scheduler", "run_due_jobs", "time.sleep(60)"),
    },
}
STARTUP_PLIST_CONTRACTS = {
    "Telegram LaunchAgent": {
        "path": Path("com.jarvis-v3.telegram.plist"),
        "label": "com.jarvis-v3.telegram",
        "module": "jarvis_v2.scripts.run_telegram_control",
        "log_filename": "telegram.log",
        "primary": True,
        "activation_environment": {
            V3_DAEMON_ENABLE_ENV: "1",
            V3_SCHEDULER_ENABLE_ENV: "0",
        },
    },
    "iMessage LaunchAgent": {
        "path": Path("com.jarvis-v3.imessage.plist"),
        "label": "com.jarvis-v3.imessage",
        "module": "jarvis_v2.scripts.run_imessage_control",
        "log_filename": "imessage.log",
        "primary": False,
        "activation_environment": {
            V3_DAEMON_ENABLE_ENV: "1",
            V3_SCHEDULER_ENABLE_ENV: "0",
        },
    },
    "status dashboard LaunchAgent": {
        "path": Path("com.jarvis-v3.dashboard.plist"),
        "label": "com.jarvis-v3.dashboard",
        "module": "jarvis_v2.scripts.run_status_server",
        "log_filename": "dashboard.log",
        "throttle_interval": 30,
        "primary": True,
        "activation_environment": {
            V3_DAEMON_ENABLE_ENV: "1",
            V3_SCHEDULER_ENABLE_ENV: "0",
        },
    },
}
DAEMON_SOURCE_FRESHNESS_CONTRACTS = {
    "Telegram": {
        "module": "jarvis_v2.scripts.run_telegram_control",
        "source_dirs": (
            Path("jarvis_v2/agent"),
            Path("jarvis_v2/automations"),
            Path("jarvis_v2/memory"),
            Path("jarvis_v2/tools"),
        ),
        "sources": (
            Path("jarvis_v2/scripts/run_telegram_control.py"),
            Path("jarvis_v2/config.py"),
            Path("jarvis_v2/env.py"),
            Path("requirements.txt"),
        ),
    },
    "dashboard": {
        "module": "jarvis_v2.scripts.run_status_server",
        "source_dirs": (
            Path("jarvis_v2/agent"),
            Path("jarvis_v2/automations"),
            Path("jarvis_v2/memory"),
            Path("jarvis_v2/tools"),
            Path("jarvis_v2/ui"),
        ),
        "sources": (
            Path("jarvis_v2/scripts/run_status_server.py"),
            Path("jarvis_v2/scripts/startup.py"),
            Path("jarvis_v2/scripts/hud.py"),
            Path("jarvis_v2/scripts/talk.py"),
            Path("jarvis_v2/config.py"),
            Path("jarvis_v2/env.py"),
            Path("requirements.txt"),
        ),
    },
}
DAEMON_SOURCE_FRESHNESS_MTIME_GRACE_SECONDS = 1.0
DAEMON_SOURCE_GIT_STATUS_MAX_CHARS = 64 * 1024
DAEMON_SOURCE_GIT_HEAD_RE = re.compile(r"[0-9a-f]{40}(?:[0-9a-f]{24})?")
STARTUP_PLIST_MAX_BYTES = 128 * 1024
STARTUP_REQUIRED_PLIST_FIELDS = frozenset(
    {
        "Label",
        "ProgramArguments",
        "WorkingDirectory",
        "EnvironmentVariables",
        "StandardOutPath",
        "StandardErrorPath",
        "RunAtLoad",
        "KeepAlive",
        "ProcessType",
    }
)
STARTUP_CRITICAL_PLIST_FIELDS = (
    "Label",
    "Program",
    "ProgramArguments",
    "WorkingDirectory",
    "EnvironmentVariables",
    "StandardOutPath",
    "StandardErrorPath",
    "RunAtLoad",
    "KeepAlive",
    "ProcessType",
    "ThrottleInterval",
)
STARTUP_ALLOWED_LOADED_ENVIRONMENT_EXTRAS = frozenset({"XPC_SERVICE_NAME"})
STARTUP_LAUNCHD_PATH = "/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin"
STARTUP_SELECTED_ENV = "JARVIS_V3_ENV"
STARTUP_LOG_DIRECTORY_RELATIVE = Path("Library") / "Logs" / "JarvisV3"
STARTUP_LOG_DIRECTORY_MODE = 0o700
STARTUP_LOG_FILE_MODE = 0o600
STARTUP_PRIVATE_RECOVERY_DOC_CONTRACTS = {
    "Telegram manual restart": ("launchctl kickstart -k gui/$(id -u)/com.jarvis-v3.telegram",),
    "iMessage manual restart": ("launchctl kickstart -k gui/$(id -u)/com.jarvis-v3.imessage",),
    "dashboard manual restart": (V3_DASHBOARD_COMMAND,),
    "dashboard LaunchAgent restart": (
        "launchctl kickstart -k gui/$(id -u)/com.jarvis-v3.dashboard",
    ),
    "dashboard authentication recovery": (
        "sign in as `jarvis` with the configured",
        "JARVIS_STATUS_AUTH_TOKEN",
        "stop the local dashboard",
        "setup_status_auth",
        "fresh unique owner-only token",
        "restart the dashboard",
        "never",
        "echoes credentials or request headers",
    ),
    "daemon logs": (
        "Library/Logs/JarvisV3/telegram.log",
        "Library/Logs/JarvisV3/imessage.log",
        "Library/Logs/JarvisV3/dashboard.log",
    ),
    "no-process-control caveat": ("never restart daemons", "Diagnostics never execute"),
    "reboot-proof caveat": ("Live reboot proof still needs operator present",),
}
STARTUP_PUBLIC_RECOVERY_BOUNDARY = (
    (
        "The sanitized public candidate contains the offline contract generator and content-free "
        "recovery receipt helpers, but no LaunchAgent templates, installed contracts, installation "
        "or restart commands, or cutover authority"
    ),
    "intentionally contains no",
    "`launchctl` or daemon-restart command",
    "Live reboot proof still needs the operator present",
    "never restart daemons",
    "Diagnostics never execute",
)


@dataclass(frozen=True)
class DaemonProcessSnapshot:
    module: str
    started_at_epoch: float
    pid: int = 0


@dataclass(frozen=True)
class DaemonSourceGitSnapshot:
    head: str
    clean: bool


@dataclass(frozen=True)
class LaunchdJobSnapshot:
    label: str
    pid: int
    state: str
    program: str = ""
    program_arguments: tuple[str, ...] = ()
    working_directory: str = ""
    daemon_enable: str | None = None
    selected_environment: str | None = None
    scheduler_enable: str | None = None
    environment: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class LaunchdJobExpectation:
    module: str
    program: str
    program_arguments: tuple[str, ...]
    working_directory: str
    daemon_enable: str
    selected_environment: str
    scheduler_enable: str
    environment: tuple[tuple[str, str], ...]

TALK_REQUIRED_TEXT = {
    "argparse --speak": ("--speak", "Speak Jarvis's replies aloud."),
    "argparse --list-mics": ("--list-mics", "List available microphones"),
    "argparse --mic": ("--mic", "Audio device index"),
    "ffmpeg capture contract": ("ffmpeg", "-f", "avfoundation", "-t", "str(_max_record_seconds())"),
    "runtime bridge": ("JarvisRuntime", "runtime.handle(transcript)", "run_turn(runtime, transcript"),
    "approval hold": ("_approval_id_from_result", "I won't act on it from voice", "Approve it in Telegram or the dashboard"),
    "speech output": ("voice.speak_text(spoken)",),
    "transcriber contract": ("voice._audio_file_transcriber()",),
    "temp clip cleanup": ("TemporaryDirectory", "clip.wav"),
}
TELEGRAM_VOICE_REQUIRED_TEXT = {
    "voice/audio detection": ('message.get("voice")', 'message.get("audio")', "_get_voice_file_id"),
    "download size cap": ("MAX_VOICE_FILE_BYTES", "20 * 1024 * 1024", "too large"),
    "telegram getFile": ('"getFile"', "FILE_API_BASE", "_download_telegram_file"),
    "temp voice file": ("TemporaryDirectory", "jarvis-tg-voice-", "voice{suffix}"),
    "local transcriber": ("voice._audio_file_transcriber()", "transcriber(path)", "Voice transcription isn't set up"),
    "safe transcript fallback": (
        "Resend a clearer voice note",
        "type the command instead",
        "`voice setup check`",
    ),
    "heard receipt": ("🎙 Heard:", "_safe_outbound_text"),
    "owner lock": ("chat_id != owner", "continue"),
    "runtime handoff": (
        "text = transcript",
        "_update_request_token(update_id)",
        "_handle_command(text, request_token=request_token)",
        "self._runtime_instance().handle(",
        "request_token=request_token",
    ),
}
ERROR_GUIDANCE_FORBIDDEN_FRAGMENTS = (
    "Traceback",
    "HTTP Error",
    "Bad Request",
    "raw backend",
    "raw calendar",
    "/\x55sers/",
    "/private/",
    "/var/folders/",
)
CALENDAR_READONLY_GUIDANCE_REQUIRED = (
    V3_CALENDAR_AUTH_READONLY_COMMAND,
    V3_CALENDAR_AUTH_TERMINAL_LOCATION,
)
CALENDAR_READONLY_GUIDANCE_FORBIDDEN = ERROR_GUIDANCE_FORBIDDEN_FRAGMENTS + (
    "google_calendar_reauth",
    "jarvis_v2.scripts.google_calendar_readonly_auth",
    "launch_jarvis_v3_calendar_auth.py full-access",
    V3_CALENDAR_AUTH_FULL_ACCESS_COMMAND,
)
AGGREGATE_SMOKE_MIN_MODULES = 110
AGGREGATE_SMOKE_PROOF_PATTERNS = (
    re.compile(
        r"python3 -m jarvis_v2\.scripts\.smoke_test_all[^\n]*green "
        r"\((?P<count>\d+)(?:/\d+| modules?)(?:,\s*(?P<duration>[0-9.]+)s)?\)",
        re.IGNORECASE,
    ),
    re.compile(
        r"All (?P<count>\d+) smoke modules passed\. Duration: (?P<duration>[0-9.]+)s",
        re.IGNORECASE,
    ),
)
AGGREGATE_SMOKE_PROOF_TIMESTAMP_RE = re.compile(
    r"\b(?P<date>20\d{2}-\d{2}-\d{2})[ T](?P<time>[0-2]\d:[0-5]\d(?::[0-5]\d)?)\s*KST\b",
    re.IGNORECASE,
)
AGGREGATE_SMOKE_SOURCE_ROOT = Path("jarvis_v2")
KST = timezone(timedelta(hours=9), name="KST")
MAX_LIVE_CHECK_CONTACT_QUERIES = 8
MAX_SCHEDULED_DELIVERY_DIAGNOSTIC_COUNT = 1_000_000
OPERATOR_WORKFLOW_EVAL_REQUIRED_TEXT = {
    "Korean Telegram approval gate": ("Korean telegram send lifecycle", "queued approval is the SUCCESS state"),
    "Korean contact lookup": ("Korean contact lookup", "가상연락처이 resolves with relation"),
    "channel health phrase": ("Channel health phrase", "what's my channel state"),
    "what broke diagnostics": ("What broke", "failed run is visible"),
    "on-demand morning brief": ("Morning brief pushed on demand", "owner phone channel"),
    "phone control shortcuts": ("Phone control shortcuts", "status / voice / voice setup"),
    "Telegram voice note bilingual send": ("Telegram voice note", "bilingual Korean/English send command"),
    "scheduler visibility": ("Scheduler visibility", "list scheduled jobs"),
    "memory write": ("Memory write", "remember ..."),
    "currency conversion": ("Currency conversion", "mocked FX seam"),
    "weather lookup": ("Weather lookup", "mocked wttr seam"),
    "markets summary": ("Markets summary in English and Korean", "markets summary"),
}
OPERATOR_WORKFLOW_EVAL_PROOF_RE = re.compile(
    r"(?:smoke_test_operator_evals[^\n]{0,240}\bgreen\b|\bgreen\b[^\n]{0,240}smoke_test_operator_evals)",
    re.IGNORECASE,
)
LIVE_CHECK_TABLE_LABELS = (
    "Jarvis config",
    "macOS Contacts",
    "send resolution",
    "call readiness",
    "voice transcription",
    "voice warmup",
    "dashboard voice",
    "local talk",
    "telegram voice",
    "image OCR",
    "research",
    "daily brief",
    "chat path",
    "mixed conversation",
    "personal integrations",
    "personal proofs",
    "personal routes",
    "morning brief job",
    "jobs freshness",
    "jobs 7-day proof",
    "daemon startup",
    "network recovery",
    "acceptance coverage",
    "acceptance gaps",
    "acceptance next",
    "aggregate smoke",
    "operator evals",
    "error guidance",
    "channel health",
    "guardrail plane",
    "proof ledger",
    "completion gate",
    "learning loop",
    "phone approvals",
    "phone control",
    "Telegram owner",
)
LIVE_CHECK_ONE_SCREEN_MAX_ROWS = 36
ACCEPTANCE_HARNESS_SECTION_ROWS = {
    "A. Conversation & research": ("research", "chat path", "mixed conversation"),
    "B. Channels": ("channel health", "send resolution", "call readiness", "Telegram owner"),
    "C. Personal integrations": ("macOS Contacts", "personal integrations", "personal proofs", "personal routes"),
    "D. Voice": ("voice transcription", "voice warmup", "dashboard voice", "local talk", "telegram voice"),
    "E. Autonomy & daily value": (
        "daily brief",
        "morning brief job",
        "jobs freshness",
        "jobs 7-day proof",
        "phone approvals",
        "phone control",
    ),
    "F. Reliability": (
        "daemon startup",
        "acceptance coverage",
        "acceptance gaps",
        "acceptance next",
        "aggregate smoke",
        "operator evals",
        "error guidance",
        "guardrail plane",
        "proof ledger",
        "completion gate",
        "learning loop",
    ),
}
ACCEPTANCE_HARNESS_ITEM_ROWS = {
    "A. Conversation & research": (
        ("`research <topic>` returns", ("research",)),
        ('"who is X" / "what is X"', ("research",)),
        ("Ordinary chat replies", ("chat path",)),
        ("A full mixed conversation", ("mixed conversation",)),
    ),
    "B. Channels": (
        ("KakaoTalk send:", ("send resolution", "channel health")),
        ("Telegram send:", ("Telegram owner", "channel health")),
        ("Instagram DM:", ("send resolution", "channel health")),
        ("iMessage send:", ("send resolution", "channel health")),
        ("Calls (phone/FaceTime", ("call readiness", "channel health")),
        ("`channel health` reflects", ("channel health",)),
    ),
    "C. Personal integrations": (
        ("Calendar reads", ("personal integrations", "personal proofs")),
        ("Calendar write cycle", ("personal integrations", "personal proofs")),
        ("Email read/search", ("personal integrations", "personal proofs")),
        ("Email send live", ("personal integrations", "personal proofs")),
        ("Reminders: one approved", ("personal integrations", "personal proofs")),
        ("Contacts lookup", ("macOS Contacts",)),
    ),
    "D. Voice": (
        ("Telegram voice note", ("voice transcription", "telegram voice")),
        ("Local push-to-talk", ("local talk",)),
        ("`talk.py --speak`", ("local talk",)),
        ("Korean voice input", ("voice transcription", "local talk", "telegram voice")),
    ),
    "E. Autonomy & daily value": (
        ("Morning Brief delivers", ("daily brief", "morning brief job")),
        ("All 8 scheduled jobs", ("jobs freshness", "jobs 7-day proof")),
        ("Approval flow works from the phone", ("phone approvals",)),
        ("`what broke` / `cockpit` / `approvals`", ("phone control",)),
    ),
    "F. Reliability": (
        ("`live_check` covers", ("acceptance coverage",)),
        ("Full smoke aggregate", ("aggregate smoke",)),
        ("Daemons (telegram control", ("daemon startup", "network recovery")),
        ("Every error message", ("error guidance",)),
    ),
}
ACCEPTANCE_NEXT_SECTION_ORDER = (
    "B. Channels",
    "C. Personal integrations",
    "D. Voice",
    "E. Autonomy & daily value",
    "A. Conversation & research",
    "F. Reliability",
)
ACCEPTANCE_NEXT_GUIDANCE = {
    "A. Conversation & research": (
        "next acceptance lane: A. Conversation & research; run the mixed conversation timing proof "
        "with operator present, record pass/fail and latency, and keep private prompts/content out of the handoff"
    ),
    "B. Channels": (
        "next acceptance lane: B. Channels; the operator should run the live channel matrix and report "
        "channel, pass/fail, approval shown, delivery result, and last visible stage/error; "
        "do not include secrets or message content"
    ),
    "C. Personal integrations": (
        "next acceptance lane: C. Personal integrations; run one operator-present approval-gated write "
        "proof at a time, then record pass/fail and the bounded audit row id without private content"
    ),
    "D. Voice": (
        "next acceptance lane: D. Voice; run one operator-present voice proof at a time, record language, "
        "route, pass/fail, and last visible stage/error without transcript content"
    ),
    "E. Autonomy & daily value": (
        "next acceptance lane: E. Autonomy & daily value; verify scheduled delivery, phone approval "
        "controls, and daily-job streaks with operator present, recording pass/fail and bounded status only"
    ),
    "F. Reliability": (
        "next acceptance lane: F. Reliability; run the aggregate/reboot/recovery acceptance proof with "
        "operator present and record pass/fail plus bounded diagnostics only"
    ),
}


def _canonical_acceptance_section(heading: str) -> str | None:
    """Return the canonical DoD section for a Markdown heading.

    Finish-plan headings may carry a human status annotation after the
    canonical title, such as ``B. Channels (BLOCKED on the operator's live tests)``.
    The acceptance counters must retain that section instead of silently
    dropping all of its checklist items.
    """
    clean_heading = str(heading or "").strip()
    for section in ACCEPTANCE_HARNESS_SECTION_ROWS:
        if clean_heading == section or clean_heading.startswith(f"{section} "):
            return section
    return None
ACCEPTANCE_NEXT_REPORT_FORMAT = (
    "; safe report fields: lane, result, approval shown if relevant, bounded evidence id or live_check row, "
    "last visible stage/error; omit secrets, message content, transcripts, contact handles, tokens, local paths, "
    "and screenshots with private text"
)
ACCEPTANCE_GAPS_FOLLOWUP = "run `acceptance next` for the prioritized operator-present proof lane and safe report format"
ACCEPTANCE_CHECKBOX_RE = re.compile(r"^- \[(?P<state>[ xX])\]\s+")
ACCEPTANCE_OFFLINE_ENGINEERING_ITEMS = {
    ("F. Reliability", "Every error message"): "error guidance",
}
ERROR_GUIDANCE_EXPECTED_LABELS = frozenset(
    {
        "Gmail credential setup guidance",
        "local reminder setup guidance",
        "Ollama chat fallback guidance",
        "Browser state guidance",
        "Worker fleet recovery guidance",
        "Audit lookup recovery guidance",
        "timezone database recovery guidance",
        "startup recovery audit storage guidance",
        "runtime trace missing-metadata guidance",
        "Local file not-found guidance",
        "Jarvis note not-found guidance",
        "Note-to-task not-found guidance",
        "Task-id not-found guidance",
        "Goal-id not-found guidance",
        "Memory-id not-found guidance",
        "Decision-id not-found guidance",
        "Preference not-found guidance",
        "telegram voice setup guidance",
        "Telegram voice media recovery guidance",
        "Telegram image OCR media recovery guidance",
        "Apple Reminders permission guidance",
        "voice recovery guidance",
        "Ollama status guidance",
        "Ollama planner fallback guidance",
        "calendar dead-token guidance",
        "calendar missing-setup guidance",
        "calendar generic recovery guidance",
        "HTTP not-found guidance",
        "HTTP service guidance",
        "HTTP timeout guidance",
        "daily brief degraded-section guidance",
        "weather connector recovery guidance",
        "news connector recovery guidance",
        "currency connector recovery guidance",
        "translation connector recovery guidance",
        "sunrise connector recovery guidance",
        "history connector recovery guidance",
        "Wikipedia connector recovery guidance",
        "research connector recovery guidance",
        "dictionary connector recovery guidance",
        "air-quality connector recovery guidance",
        "holidays connector recovery guidance",
        "markets connector recovery guidance",
        "joke connector recovery guidance",
        "compose-and-write model guidance",
        "writer type recovery guidance",
        "writer paste recovery guidance",
        "OCR unavailable guidance",
        "OCR read recovery guidance",
        "Telegram owner-channel recovery guidance",
        "Telegram Web permission guidance",
        "KakaoTalk send recovery guidance",
        "Instagram DM recovery guidance",
        "iMessage send recovery guidance",
        "scheduler execution and delivery recovery guidance",
    }
)
ERROR_GUIDANCE_EXPECTED_CONTRACT_COUNT = 55


def _plain_text(value: object) -> str:
    if value is None:
        return ""
    try:
        return str(value)
    except Exception:
        return ""


def _detail_text(value: object) -> str:
    if value is None:
        return ""
    try:
        return str(value)
    except Exception:
        return f"<unprintable {type(value).__name__}>"


def _safe_detail(value: object, *, limit: int = MAX_DIAGNOSTIC_CHARS) -> str:
    text = " ".join(LOCAL_PATH_RE.sub("<local-path>", _detail_text(value)).split())
    if len(text) > limit:
        return text[: limit - 1].rstrip() + "…"
    return text


def _safe_exception(exc: BaseException) -> str:
    detail = _safe_detail(exc)
    if detail:
        return f"{type(exc).__name__}: {detail}"
    return type(exc).__name__


def _handle_kind(value: object) -> str:
    text = _plain_text(value).strip()
    if not text:
        return "no phone/email handle"
    if "@" in text:
        return "email handle"
    if sum(ch.isdigit() for ch in text) >= 7:
        return "phone handle"
    return "handle"


def _safe_iso_datetime(value: object) -> str:
    try:
        parsed = datetime.fromisoformat(_plain_text(value).strip())
    except (TypeError, ValueError):
        return "invalid_timestamp"
    return parsed.replace(microsecond=0).isoformat()


def _parse_safe_datetime(value: object) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(_plain_text(value).strip())
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone().replace(tzinfo=None)
    return parsed.replace(microsecond=0)


def _safe_positive_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = int(value)
    except Exception:
        return None
    return parsed if parsed > 0 else None


def _safe_nonnegative_float(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except Exception:
        return None
    if not math.isfinite(parsed) or parsed < 0:
        return None
    return round(parsed, 1)


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return round(ordered[0], 1)
    rank = (len(ordered) - 1) * percentile
    lower = int(math.floor(rank))
    upper = int(math.ceil(rank))
    if lower == upper:
        return round(ordered[lower], 1)
    weight = rank - lower
    return round(ordered[lower] + ((ordered[upper] - ordered[lower]) * weight), 1)


def _mapping_value(row: object, key: str, default: object = "") -> object:
    try:
        if isinstance(row, dict):
            return row.get(key, default)
        return row[key]  # type: ignore[index]
    except Exception:
        return default


def _job_value(row: object, key: str, default: object = "") -> object:
    return _mapping_value(row, key, default)


def _metadata_value(metadata: object, key: str, default: object = None) -> object:
    return _mapping_value(metadata, key, default)


def _message_metadata(row: object) -> dict[str, object]:
    raw = _mapping_value(row, "metadata", {})
    if isinstance(raw, dict):
        return raw
    try:
        parsed = json.loads(_plain_text(raw) or "{}")
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _reminder_scheduled_proof(metadata: dict[str, object]) -> bool:
    handoff = _metadata_value(metadata, "set_reminder_handoff", {})
    if isinstance(handoff, dict):
        status = _plain_text(_metadata_value(handoff, "status")).strip().casefold()
        if status == "scheduled":
            return True

    status = _plain_text(_metadata_value(metadata, "status")).strip().casefold()
    stored = _metadata_value(metadata, "stored")
    if status == "scheduled":
        return True
    return stored is True or _plain_text(stored).strip().casefold() in {"1", "true", "yes"}


def _job_metadata(row: object) -> dict[str, object]:
    raw = _job_value(row, "metadata", {})
    if isinstance(raw, dict):
        return raw
    try:
        parsed = json.loads(_plain_text(raw) or "{}")
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _row_bool(row: object, key: str) -> bool:
    value = _mapping_value(row, key, False)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value != 0
    if isinstance(value, str):
        return value.strip().casefold() in {"1", "true", "yes", "ok"}
    return False


def _safe_chat_source(value: object) -> str:
    text = _plain_text(value).strip().casefold()
    if re.fullmatch(r"[a-z0-9_.-]{1,40}", text):
        return text
    return "unknown"


def _chat_latency_ms(response: object) -> float | None:
    for key in CHAT_LATENCY_METADATA_KEYS:
        parsed = _safe_nonnegative_float(_metadata_value(response, key))
        if parsed is not None:
            return parsed
    return None


def _runtime_trace(metadata: object) -> dict[str, object]:
    trace = _metadata_value(metadata, "runtime_trace", {})
    return trace if isinstance(trace, dict) else {}


def _planned_actions(trace: object) -> list[dict[str, object]]:
    if not isinstance(trace, dict):
        return []
    raw_actions = trace.get("planned_actions")
    if not isinstance(raw_actions, list):
        return []
    return [action for action in raw_actions if isinstance(action, dict)]


def _ordinary_chat_response(metadata: object) -> dict[str, object] | None:
    if _metadata_value(metadata, "runtime_route") != "chat":
        return None
    response = _metadata_value(metadata, "chat_response", {})
    if not isinstance(response, dict):
        return None
    if _metadata_value(response, "runtime_route") == "command_suggestion":
        return None
    if _metadata_value(response, "pre_planner_command_suggestion") is True:
        return None
    return response


def _mixed_action_lanes(action: dict[str, object]) -> set[str]:
    tool_name = _plain_text(action.get("tool_name")).strip()
    toolset = _plain_text(action.get("toolset")).strip().casefold()
    lanes: set[str] = set()
    if tool_name in MIXED_RESEARCH_TOOLS or (toolset == "browser" and tool_name in MIXED_RESEARCH_TOOLS):
        lanes.add("research")
    if tool_name in MIXED_CALENDAR_TOOLS:
        lanes.add("calendar")
    if tool_name in MIXED_TASK_TOOLS or toolset == "tasks":
        lanes.add("tasks")
    return lanes


def _job_enabled(row: object) -> bool:
    value = _job_value(row, "enabled", False)
    if isinstance(value, str):
        return value.strip().casefold() not in {"", "0", "false", "no", "off"}
    try:
        return bool(value)
    except Exception:
        return False


def _safe_count(value: object) -> str:
    if isinstance(value, bool):
        return "unknown"
    if isinstance(value, int) and value >= 0:
        return str(value)
    return "unknown"


def _voice_source_label(source: object) -> str:
    return VOICE_TRANSCRIBER_LABELS.get(_plain_text(source), "unknown transcriber source")


def _load_config_for_check():
    try:
        return load_config(), None
    except Exception as e:
        return None, f"config unavailable: {_safe_exception(e)}"


def _arg_contacts(argv: list[str]) -> tuple[str, ...]:
    contacts: list[str] = []
    for i, token in enumerate(argv):
        if token in ("--contact", "-c") and i + 1 < len(argv):
            contacts.append(argv[i + 1].strip())
        elif token.startswith("--contact="):
            contacts.append(token.split("=", 1)[1].strip())
    if not contacts:
        contacts.append(os.getenv("JARVIS_LIVE_CHECK_CONTACT", "").strip())

    unique: list[str] = []
    seen: set[str] = set()
    for contact in contacts:
        clean = unicodedata.normalize("NFKC", contact).strip()
        if not clean:
            continue
        identity = clean.casefold()
        if identity in seen:
            continue
        seen.add(identity)
        unique.append(clean)
    return tuple(unique)


def _arg_contact(argv: list[str]) -> str:
    """Backward-compatible single-contact accessor for external callers."""
    contacts = _arg_contacts(argv)
    return contacts[0] if contacts else ""


def _row(label: str, ok: bool | None, detail: str) -> int:
    if ok is None:
        mark = "– skip"
    else:
        mark = "✓ ok" if ok else "✗ FAIL"
    print(f"{label:<20} {mark:<8} {_safe_detail(detail, limit=50)}")
    return 1 if ok else 0


def _check_contacts(contact: str) -> tuple[bool | None, str]:
    if not contact:
        return None, "pass --contact NAME (or JARVIS_LIVE_CHECK_CONTACT) to test"
    try:
        from jarvis_v2.tools import contacts_connector as cc

        cc.clear_contact_cache()
        matches = cc.resolve_contact(contact)
    except Exception as e:  # never raise — diagnostic only
        return False, f"EXCEPTION: {_safe_exception(e)}"
    if not matches:
        try:
            store_status = cc.contacts_store_status()
        except Exception:
            return False, "couldn't access macOS Contacts — open Contacts once and grant Contacts permission to Python"
        if store_status == "empty":
            return False, "macOS Contacts has no saved contacts — sync the required contacts to this Mac, then retry"
        if store_status == "unavailable":
            return False, "couldn't access macOS Contacts — open Contacts once and grant Contacts permission to Python"
        return False, "no match for supplied contact query — try a name you know exists in macOS Contacts"
    handle_kind = _handle_kind(matches[0].handle)
    return True, f"{len(matches)} match(es); primary match has {handle_kind}"


def _check_send_resolution(contact: str) -> tuple[bool | None, str]:
    """Pre-flight: what would `text <contact>` actually do before any real send?

    Mirrors the send tools' decision (handle bypass / single resolve / ambiguous
    block / not found) read-only, so the first live send isn't a blind guess.
    """
    if not contact:
        return None, "pass --contact NAME to preview the send decision"
    try:
        from jarvis_v2.tools import contacts_connector as cc

        if cc.looks_like_handle(contact):
            return True, f"already a {_handle_kind(contact)} — would send directly"
        cc.clear_contact_cache()
        matches = cc.resolve_contact(contact)
    except Exception as e:  # never raise — diagnostic only
        return False, f"EXCEPTION: {_safe_exception(e)}"
    if not matches:
        return False, "would NOT send — no contact found for supplied query"
    if len(matches) > 1:
        return None, f"would BLOCK & ask which — {len(matches)} matching contacts"
    m = matches[0]
    if not m.handle:
        return False, "would NOT send — single matched contact has no phone/email on file"
    return True, f"would send to single matched contact via {_handle_kind(m.handle)}"


def _check_native_call_target(contact: str) -> tuple[bool | None, str]:
    """Read-only native-call target resolution using the approval-time rules."""

    if not contact:
        return None, "pass --contact NAME to bind the native phone/FaceTime target"
    try:
        from jarvis_v2.tools import call_connector
        from jarvis_v2.tools import contacts_connector as cc

        if cc.looks_like_handle(contact):
            if call_connector.call_handle_is_valid("phone", contact):
                return True, "explicit phone handle is valid; exact value suppressed"
            return False, "phone-call target is not a phone handle; request FaceTime or supply a phone number"
        cc.clear_contact_cache()
        matches = cc.resolve_contact(contact)
    except Exception as e:
        return False, f"EXCEPTION: {_safe_exception(e)}"
    if not matches:
        return False, "native call would NOT queue approval — no contact found"
    if len(matches) > 1:
        return None, f"native call would BLOCK before approval — {len(matches)} matching contacts"
    phone_handle = call_connector.call_handle_for_mode(matches[0], "phone")
    facetime_handle = call_connector.call_handle_for_mode(matches[0], "audio")
    if not phone_handle:
        if facetime_handle:
            return False, "single contact is FaceTime-only; phone-call acceptance target lacks a phone number"
        return False, "single contact has no usable phone/FaceTime handle"
    if not call_connector.call_handle_is_valid("phone", phone_handle):
        return False, "single contact resolved to an invalid phone handle"
    if not facetime_handle or not call_connector.call_handle_is_valid("audio", facetime_handle):
        return False, "single contact has no valid FaceTime handle"
    return True, "single contact has exact phone + FaceTime target binding; handle suppressed"


def _check_call_readiness(config, contacts: tuple[str, ...]) -> tuple[bool | None, str]:
    """Read-only call gate/target preflight; never opens an app or invokes a call handler."""

    expected_fields = {
        "call_contact": {"to", "recipient", "contact", "name", "mode", "kind", "type"},
        "call_kakao": {"to", "recipient", "contact", "name", "mode", "kind", "type"},
        "call_instagram": {"to", "recipient", "contact", "name", "mode", "kind", "type"},
        "call_telegram": {"to", "recipient", "name", "mode"},
    }
    expected = set(expected_fields)
    try:
        from jarvis_v2.agent.types import RiskLevel
        from jarvis_v2.tools.call_connector import make_call_tools
        from jarvis_v2.tools.registry import ToolArgumentType

        tools = {tool.name: tool for tool in make_call_tools(config)}
    except Exception as e:
        return False, f"call readiness unavailable: {_safe_exception(e)}"
    missing = sorted(expected.difference(tools))
    wrong_risk = sorted(
        name for name in expected.intersection(tools) if tools[name].risk != RiskLevel.HIGH_RISK
    )
    if missing or wrong_risk:
        return False, (
            f"call gate drift: {len(missing)} missing, {len(wrong_risk)} not HIGH_RISK — "
            "repair tool registration before live proof"
        )
    strict_argument_drift = []
    for name in sorted(expected):
        contract = tools[name].argument_contract
        fields = tuple(contract.fields) if contract is not None else ()
        if (
            contract is None
            or contract.allow_unknown is not False
            or {field.name for field in fields} != expected_fields[name]
            or any(field.required for field in fields)
            or any(field.types != frozenset({ToolArgumentType.STRING}) for field in fields)
        ):
            strict_argument_drift.append(name)
    if strict_argument_drift:
        return False, (
            f"call argument-contract drift: {len(strict_argument_drift)}/4 tools are not strict typed bindings — "
            "repair tool registration before live proof"
        )

    base = (
        "4/4 call tools HIGH_RISK, strict-argument, and approval-gated; "
        "the approval card must bind the exact target + effective mode"
    )
    live_boundary = (
        "Kakao/Instagram/Telegram account and call-button availability still need operator-present proof; "
        "call_requested is not recipient-confirmed proof; known-not-started and outcome-unknown are distinct; "
        "an outcome-unknown call must be checked and never retried automatically; "
        "no call, approval, browser, clipboard, or app control ran"
    )
    if not contacts:
        return None, (
            f"{base}; pass --contact NAME to bind the native phone/FaceTime target; {live_boundary}"
        )
    if len(contacts) > MAX_LIVE_CHECK_CONTACT_QUERIES:
        return False, (
            f"call target preflight limit is {MAX_LIVE_CHECK_CONTACT_QUERIES}; "
            "rerun with fewer supplied names"
        )

    results = [_check_native_call_target(contact)[0] for contact in contacts]
    ready = sum(result is True for result in results)
    failed = sum(result is False for result in results)
    unproven = sum(result is None for result in results)
    total = len(contacts)
    detail = (
        f"{base}; {ready}/{total} native phone/FaceTime target binding(s) ready; "
        "supplied names and handles hidden"
    )
    if failed:
        return False, f"{detail}; {failed} need contact repair; {live_boundary}"
    if unproven:
        return None, f"{detail}; {unproven} need disambiguation; {live_boundary}"
    return True, f"{detail}; {live_boundary}"


def _check_contact_matrix(
    contacts: tuple[str, ...],
    checker: Callable[[str], tuple[bool | None, str]],
    *,
    label: str,
) -> tuple[bool | None, str]:
    """Run bounded named-contact preflights without rendering supplied names."""
    if not contacts:
        return checker("")
    if len(contacts) > MAX_LIVE_CHECK_CONTACT_QUERIES:
        return (
            False,
            f"contact preflight limit is {MAX_LIVE_CHECK_CONTACT_QUERIES}; rerun with fewer supplied names",
        )
    if len(contacts) == 1:
        return checker(contacts[0])

    results = [checker(contact)[0] for contact in contacts]
    ready = sum(result is True for result in results)
    failed = sum(result is False for result in results)
    unproven = sum(result is None for result in results)
    total = len(contacts)
    if failed:
        return False, f"{ready}/{total} {label} ready; {failed} need attention; supplied names hidden"
    if unproven:
        return None, f"{ready}/{total} {label} ready; {unproven} need disambiguation; supplied names hidden"
    return True, f"{ready}/{total} {label} ready; supplied names hidden"


def _check_voice_transcription() -> tuple[bool | None, str]:
    """Read-only: is speech transcription set up (for talk + Telegram voice memos)?"""
    try:
        from jarvis_v2.tools import voice

        source = voice._voice_local_transcriber_source()
    except Exception as e:
        return False, f"EXCEPTION: {_safe_exception(e)}"
    if source == "not_configured":
        return False, "no transcriber — install the `whisper` CLI or set JARVIS_VOICE_WHISPER_MODEL_PATH"
    detail = _voice_source_label(source)
    if detail == "unknown transcriber source":
        return False, detail
    if source == "local_whisper_cli":
        try:
            from jarvis_v2.tools import voice as _v

            if _v._whisper_cli_path():
                detail += " found"
        except Exception:
            pass
    return True, f"ready via {detail}"


def _check_voice_warmup() -> tuple[bool | None, str]:
    """Read-only: is the opt-in first-transcription warmup ready if enabled?"""
    try:
        from jarvis_v2.tools import voice

        enabled = voice._voice_warmup_enabled()
        source = voice._voice_local_transcriber_source()
    except Exception as e:
        return False, f"EXCEPTION: {_safe_exception(e)}"
    if not enabled:
        return None, "disabled — set JARVIS_VOICE_WARMUP=1 to prewarm the local transcriber"
    if source == "not_configured":
        return False, "enabled but no local audio transcriber is configured"
    detail = _voice_source_label(source)
    if detail == "unknown transcriber source":
        return False, "enabled but transcriber source is unknown"
    return True, f"enabled; will warm via {detail} using generated silence, no microphone"


def _check_dashboard_voice_wiring() -> tuple[bool | None, str]:
    """Read-only: does the dashboard reuse the mini-HUD voice pipeline?"""
    try:
        from jarvis_v2.scripts import hud
        from jarvis_v2.ui import status_server
    except Exception as e:
        return False, f"EXCEPTION: {_safe_exception(e)}"
    recorder_shared = status_server.HudRecorder is hud.HudRecorder
    stop_shared = status_server.stop_speech is hud.stop_speech
    if recorder_shared and stop_shared:
        return True, "dashboard reuses HudRecorder + stop_speech from mini-HUD"
    missing = []
    if not recorder_shared:
        missing.append("HudRecorder")
    if not stop_shared:
        missing.append("stop_speech")
    return False, "dashboard voice drift: " + ", ".join(missing)


def _check_local_talk_readiness(root: Path | None = None) -> tuple[bool | None, str]:
    """Read-only: does local push-to-talk keep the expected safety contract?"""
    root_path = root or REPO_ROOT
    path = root_path / "jarvis_v2" / "scripts" / "talk.py"
    try:
        text = path.read_text(encoding="utf-8")
    except Exception as e:
        return False, f"talk.py unreadable: {_safe_exception(e)}"

    missing = [
        label
        for label, tokens in TALK_REQUIRED_TEXT.items()
        if any(token not in text for token in tokens)
    ]
    try:
        from jarvis_v2.scripts import talk

        max_record_default = talk._max_record_seconds()
    except Exception as e:
        return False, f"talk.py import/readiness failed: {_safe_exception(e)}"
    if max_record_default != 120:
        missing.append("bounded default recording cap")

    if missing:
        return (
            False,
            f"talk readiness drift {len(missing)} contract(s): {', '.join(missing[:6])} — repair local push-to-talk before live mic proof",
        )
    if not shutil.which("ffmpeg"):
        return (
            False,
            "ffmpeg is unavailable — install it, then rerun local talk readiness before the live mic proof; "
            "no microphone, transcription, runtime turn, speech, or account access run",
        )

    return (
        True,
        (
            f"ready: {len(TALK_REQUIRED_TEXT)} static contract(s); --speak/--list-mics/--mic present; "
            f"default recording cap {max_record_default}s; voice approvals stay held for Telegram/dashboard; "
            "ffmpeg prerequisite found; no microphone, transcription, runtime turn, speech, or account access run"
        ),
    )


def _check_telegram_voice_readiness(root: Path | None = None) -> tuple[bool | None, str]:
    """Read-only: does Telegram voice-note intake keep the expected safety contract?"""
    root_path = root or REPO_ROOT
    path = root_path / "jarvis_v2" / "automations" / "telegram_control.py"
    try:
        text = path.read_text(encoding="utf-8")
    except Exception as e:
        return False, f"telegram voice source unreadable: {_safe_exception(e)}"

    missing = [
        label
        for label, tokens in TELEGRAM_VOICE_REQUIRED_TEXT.items()
        if any(token not in text for token in tokens)
    ]
    if root is None:
        try:
            from jarvis_v2.automations import telegram_control

            if telegram_control.MAX_VOICE_FILE_BYTES != 20 * 1024 * 1024:
                missing.append("download size cap value")
            if (
                telegram_control._get_voice_file_id(
                    {"voice": {"file_id": "ok", "file_size": telegram_control.MAX_VOICE_FILE_BYTES}}
                )
                != "ok"
            ):
                missing.append("voice file_id extraction")
            if (
                telegram_control._get_voice_file_id(
                    {"audio": {"file_id": "ok-audio", "file_size": 1024}}
                )
                != "ok-audio"
            ):
                missing.append("audio file_id extraction")
            if (
                telegram_control._get_voice_file_id(
                    {
                        "voice": {
                            "file_id": "too-large",
                            "file_size": telegram_control.MAX_VOICE_FILE_BYTES + 1,
                        }
                    }
                )
                != ""
            ):
                missing.append("oversized file rejection")
        except Exception as e:
            return False, f"telegram voice helper check failed: {_safe_exception(e)}"

    if missing:
        highlighted: list[str] = []
        for label in missing[:4] + [
            label
            for label in ("safe transcript fallback", "owner lock", "runtime handoff")
            if label in missing
        ]:
            if label not in highlighted:
                highlighted.append(label)
        return (
            False,
            f"telegram voice readiness drift {len(missing)} contract(s): {', '.join(highlighted[:8])} — repair phone voice intake before live proof",
        )

    return (
        True,
        (
            f"ready: {len(TELEGRAM_VOICE_REQUIRED_TEXT)} static contract(s); voice/audio notes use a 20MB cap, "
            "Telegram getFile download, temp voice file cleanup, local transcriber, safe transcript fallback, "
            "heard receipt, owner lock, and the same approval-gated runtime handoff; no Telegram polling/download, "
            "microphone, transcription, runtime turn, account access, or send run"
        ),
    )


def _guidance_problem(
    label: str,
    message: object,
    *,
    required: tuple[str, ...],
    forbidden: tuple[str, ...] = ERROR_GUIDANCE_FORBIDDEN_FRAGMENTS,
) -> str | None:
    if isinstance(message, _ToolResultGuidance):
        output_required = tuple(fragment for fragment in required if not fragment.startswith('"'))
        metadata_required = tuple(fragment for fragment in required if fragment.startswith('"'))
        output_problem = _guidance_problem(
            f"{label} visible output",
            message.output,
            required=output_required,
            forbidden=forbidden,
        )
        if output_problem:
            return output_problem
        if metadata_required:
            metadata_problem = _guidance_problem(
                f"{label} recovery metadata",
                message.metadata,
                required=metadata_required,
                forbidden=forbidden,
            )
            if metadata_problem:
                return metadata_problem
        if message.per_result_required:
            if len(message.entries) != len(message.per_result_required):
                return f"{label} per-result recovery contract count changed"
            for index, ((output, metadata), (entry_output_required, entry_metadata_required)) in enumerate(
                zip(message.entries, message.per_result_required),
                start=1,
            ):
                entry_output_problem = _guidance_problem(
                    f"{label} result {index} visible output",
                    output,
                    required=entry_output_required,
                    forbidden=forbidden,
                )
                if entry_output_problem:
                    return entry_output_problem
                entry_metadata_problem = _guidance_problem(
                    f"{label} result {index} recovery metadata",
                    metadata,
                    required=entry_metadata_required,
                    forbidden=forbidden,
                )
                if entry_metadata_problem:
                    return entry_metadata_problem
        return None
    text = _plain_text(message).strip()
    lower = text.lower()
    if not text:
        return f"{label} returned empty guidance"
    missing = [fragment for fragment in required if fragment.lower() not in lower]
    if missing:
        return f"{label} missing {', '.join(missing[:2])}"
    if any(fragment.lower() in lower for fragment in forbidden):
        return f"{label} leaked raw detail"
    return None


class _ToolResultGuidance:
    def __init__(
        self,
        output: str,
        metadata: str,
        entries: tuple[tuple[str, str], ...],
        per_result_required: tuple[
            tuple[tuple[str, ...], tuple[str, ...]], ...
        ],
    ) -> None:
        self.output = output
        self.metadata = metadata
        self.entries = entries
        self.per_result_required = per_result_required


def _tool_result_guidance(
    *results: object,
    per_result_required: tuple[
        tuple[tuple[str, ...], tuple[str, ...]], ...
    ] = (),
) -> _ToolResultGuidance:
    output_parts: list[str] = []
    metadata_parts: list[str] = []
    entries: list[tuple[str, str]] = []
    for result in results:
        output = _plain_text(getattr(result, "output", "")).strip()
        if output:
            output_parts.append(output)
        metadata = getattr(result, "metadata", None)
        metadata_text = ""
        if isinstance(metadata, dict):
            metadata_text = json.dumps(
                metadata,
                ensure_ascii=False,
                sort_keys=True,
                default=lambda value: type(value).__name__,
            )
            metadata_parts.append(metadata_text)
        entries.append((output, metadata_text))
    return _ToolResultGuidance(
        "\n".join(output_parts),
        "\n".join(metadata_parts),
        tuple(entries),
        per_result_required,
    )


def _error_guidance_inventory_problem(labels: tuple[str, ...]) -> str | None:
    actual = set(labels)
    duplicates = sorted({label for label in labels if labels.count(label) > 1})
    missing = sorted(ERROR_GUIDANCE_EXPECTED_LABELS - actual)
    unexpected = sorted(actual - ERROR_GUIDANCE_EXPECTED_LABELS)
    if len(labels) != ERROR_GUIDANCE_EXPECTED_CONTRACT_COUNT:
        return (
            "recovery coverage count changed "
            f"({len(labels)}/{ERROR_GUIDANCE_EXPECTED_CONTRACT_COUNT}); review the acceptance contract"
        )
    if duplicates or missing or unexpected:
        parts: list[str] = []
        if duplicates:
            parts.append(f"duplicate recovery contract(s): {', '.join(duplicates[:2])}")
        if missing:
            parts.append(f"missing recovery contract(s): {', '.join(missing[:2])}")
        if unexpected:
            parts.append(f"unexpected recovery contract(s): {', '.join(unexpected[:2])}")
        return "; ".join(parts)
    return None


@dataclass(frozen=True)
class AggregateSmokeProof:
    module_count: int
    duration: str | None
    recorded_at: datetime | None


def _aggregate_smoke_proof_timestamp(line: str) -> datetime | None:
    match = AGGREGATE_SMOKE_PROOF_TIMESTAMP_RE.search(line)
    if match is None:
        return None
    try:
        return datetime.fromisoformat(f"{match.group('date')}T{match.group('time')}").replace(tzinfo=KST)
    except ValueError:
        return None


def _latest_aggregate_smoke_proof(text: str) -> AggregateSmokeProof | None:
    # The maintenance log is normally prepended, but handoff edits can place a
    # newer proof below an older entry. Prefer explicit KST timestamps over
    # document position so freshness does not depend on formatting discipline.
    newest_timestamped: AggregateSmokeProof | None = None
    first_untimestamped: AggregateSmokeProof | None = None
    for line in text.splitlines():
        if "smoke_test_all" not in line and "smoke modules passed" not in line:
            continue
        for pattern in AGGREGATE_SMOKE_PROOF_PATTERNS:
            match = pattern.search(line)
            if match:
                proof = AggregateSmokeProof(
                    module_count=int(match.group("count")),
                    duration=match.groupdict().get("duration"),
                    recorded_at=_aggregate_smoke_proof_timestamp(line),
                )
                if proof.recorded_at is None:
                    if first_untimestamped is None:
                        first_untimestamped = proof
                elif newest_timestamped is None or proof.recorded_at > newest_timestamped.recorded_at:
                    newest_timestamped = proof
                break
    return newest_timestamped or first_untimestamped


def _latest_aggregate_smoke_source_mtime(root: Path) -> float | None:
    """Return the newest local Python source mtime used by the aggregate suite."""
    source_root = root / AGGREGATE_SMOKE_SOURCE_ROOT
    try:
        mtimes = [
            path.stat().st_mtime
            for path in source_root.rglob("*.py")
            if path.is_file() and "__pycache__" not in path.parts
        ]
    except OSError:
        return None
    return max(mtimes, default=None)


def _check_aggregate_smoke_proof(root: Path | None = None) -> tuple[bool | None, str]:
    """Read-only: is there a logged aggregate smoke proof for the cockpit?"""
    tasks_path = (root or REPO_ROOT) / "CODEX_TASKS.md"
    try:
        tasks_text = tasks_path.read_text(encoding="utf-8")
    except Exception as e:
        return False, f"aggregate smoke proof unreadable: {_safe_exception(e)}"

    proof = _latest_aggregate_smoke_proof(tasks_text)
    if proof is None:
        return (
            False,
            "aggregate smoke proof missing: no logged smoke_test_all green proof found; run the aggregate suite before DONE",
        )

    if proof.module_count < AGGREGATE_SMOKE_MIN_MODULES:
        return (
            False,
            (
                f"aggregate smoke proof below target: logged {proof.module_count} module(s), "
                f"expected at least {AGGREGATE_SMOKE_MIN_MODULES}; rerun the aggregate suite before DONE"
            ),
        )

    if proof.recorded_at is None:
        return (
            None,
            "aggregate smoke proof timestamp missing: log a KST timestamp with the green aggregate suite before DONE",
        )

    source_mtime = _latest_aggregate_smoke_source_mtime(tasks_path.parent)
    if source_mtime is None:
        return (
            None,
            "aggregate smoke source freshness unavailable: repair the local source tree, then rerun the aggregate suite before DONE",
        )
    if proof.recorded_at.timestamp() < source_mtime:
        return (
            False,
            "aggregate smoke proof predates current source: rerun the aggregate suite before DONE",
        )

    duration_detail = f", duration {proof.duration}s" if proof.duration else ""
    return (
        True,
        (
            f"aggregate smoke proof ready: latest logged smoke_test_all green proof reports {proof.module_count} modules"
            f"{duration_detail}; read-only log check only, did not run tests"
        ),
    )


def _latest_operator_workflow_eval_proof(text: str) -> bool:
    for line in text.splitlines():
        if "smoke_test_operator_evals" not in line:
            continue
        if OPERATOR_WORKFLOW_EVAL_PROOF_RE.search(line):
            return True
    return False


def _check_operator_workflow_eval_proof(root: Path | None = None) -> tuple[bool | None, str]:
    """Read-only: is the operator's real-workflow eval pack present and recently proven green?"""
    root_path = root or REPO_ROOT
    eval_path = root_path / "jarvis_v2" / "scripts" / "smoke_test_operator_evals.py"
    tasks_path = root_path / "CODEX_TASKS.md"
    try:
        eval_text = eval_path.read_text(encoding="utf-8")
    except Exception as e:
        return False, f"operator workflow eval source unreadable: {_safe_exception(e)}"
    try:
        tasks_text = tasks_path.read_text(encoding="utf-8")
    except Exception as e:
        return False, f"operator workflow eval proof unreadable: {_safe_exception(e)}"

    missing = [
        label
        for label, tokens in OPERATOR_WORKFLOW_EVAL_REQUIRED_TEXT.items()
        if any(token not in eval_text for token in tokens)
    ]
    if missing:
        return (
            False,
            (
                f"operator workflow eval pack drift: {len(missing)} contract(s) missing; "
                f"repair smoke_test_operator_evals before claiming real-workflow coverage"
            ),
        )

    if not _latest_operator_workflow_eval_proof(tasks_text):
        return (
            False,
            (
                "operator workflow eval proof missing: no logged smoke_test_operator_evals green proof found; "
                "run the focused eval pack before DONE"
            ),
        )

    return (
        True,
        (
            f"operator workflow eval proof ready: {len(OPERATOR_WORKFLOW_EVAL_REQUIRED_TEXT)} operator-real workflow lane(s) "
            "covered by smoke_test_operator_evals; latest logged focused proof is green; read-only source/log check only, "
            "did not run evals, send, call, approve, fetch, transcribe, or use real contacts"
        ),
    )


def _check_acceptance_harness_coverage(root: Path | None = None) -> tuple[bool | None, str]:
    """Read-only: does live_check still map onto the August DoD sections?"""
    plan_path = (root or REPO_ROOT) / "FINISH_PLAN_AUGUST.md"
    try:
        plan_text = plan_path.read_text(encoding="utf-8")
    except Exception as e:
        return False, f"acceptance coverage plan unreadable: {_safe_exception(e)}"

    labels = set(LIVE_CHECK_TABLE_LABELS)
    duplicate_rows = len(LIVE_CHECK_TABLE_LABELS) - len(labels)
    plan_sections = {
        canonical
        for line in plan_text.splitlines()
        if line.startswith("### ")
        for canonical in (_canonical_acceptance_section(line[4:]),)
        if canonical is not None
    }
    missing_sections = [section for section in ACCEPTANCE_HARNESS_SECTION_ROWS if section not in plan_sections]
    missing_rows: list[str] = []
    for section, rows in ACCEPTANCE_HARNESS_SECTION_ROWS.items():
        for row in rows:
            if row not in labels:
                missing_rows.append(section)

    checklist_lines = {section: [] for section in ACCEPTANCE_HARNESS_SECTION_ROWS}
    current_section: str | None = None
    for line in plan_text.splitlines():
        if line.startswith("### "):
            current_section = _canonical_acceptance_section(line[4:])
            continue
        if current_section is not None and ACCEPTANCE_CHECKBOX_RE.match(line):
            checklist_lines[current_section].append(line)

    expected_sections = set(ACCEPTANCE_HARNESS_SECTION_ROWS)
    item_map_sections = set(ACCEPTANCE_HARNESS_ITEM_ROWS)
    item_map_section_drift = sorted(expected_sections.symmetric_difference(item_map_sections))
    item_count_mismatches: list[str] = []
    item_anchor_mismatches: list[str] = []
    item_row_mismatches: list[str] = []
    duplicate_item_anchors: list[str] = []
    mapped_item_count = 0
    plan_item_count = sum(len(lines) for lines in checklist_lines.values())
    for section in expected_sections.intersection(item_map_sections):
        mappings = ACCEPTANCE_HARNESS_ITEM_ROWS[section]
        lines = checklist_lines[section]
        if len(lines) != len(mappings):
            item_count_mismatches.append(section)
        anchors = [anchor for anchor, _rows in mappings]
        if len(anchors) != len(set(anchors)):
            duplicate_item_anchors.append(section)
        for anchor, rows in mappings:
            if not anchor or sum(1 for line in lines if anchor in line) != 1:
                item_anchor_mismatches.append(section)
            if not rows or any(row not in labels for row in rows):
                item_row_mismatches.append(section)
        for line in lines:
            matches = [anchor for anchor in anchors if anchor and anchor in line]
            if len(matches) != 1:
                item_anchor_mismatches.append(section)
            else:
                mapped_item_count += 1

    if (
        missing_sections
        or missing_rows
        or duplicate_rows
        or item_map_section_drift
        or item_count_mismatches
        or item_anchor_mismatches
        or item_row_mismatches
        or duplicate_item_anchors
        or len(LIVE_CHECK_TABLE_LABELS) > LIVE_CHECK_ONE_SCREEN_MAX_ROWS
    ):
        problems = []
        if missing_sections:
            problems.append(f"missing DoD section(s): {', '.join(missing_sections[:3])}")
        if missing_rows:
            problems.append(f"missing live_check row mapping(s): {', '.join(sorted(set(missing_rows))[:3])}")
        if item_map_section_drift:
            problems.append(f"item-map section drift: {', '.join(item_map_section_drift[:3])}")
        if item_count_mismatches:
            problems.append(
                f"checklist item count mismatch(s): {', '.join(sorted(set(item_count_mismatches))[:3])}"
            )
        if item_anchor_mismatches:
            problems.append(
                f"checklist item anchor mismatch(s): {', '.join(sorted(set(item_anchor_mismatches))[:3])}"
            )
        if item_row_mismatches:
            problems.append(
                f"checklist item row mapping(s) missing: {', '.join(sorted(set(item_row_mismatches))[:3])}"
            )
        if duplicate_item_anchors:
            problems.append(
                f"duplicate checklist item anchor(s): {', '.join(sorted(set(duplicate_item_anchors))[:3])}"
            )
        if duplicate_rows:
            problems.append(f"{duplicate_rows} duplicate live_check row label(s)")
        if len(LIVE_CHECK_TABLE_LABELS) > LIVE_CHECK_ONE_SCREEN_MAX_ROWS:
            problems.append(
                f"one-screen row budget exceeded: {len(LIVE_CHECK_TABLE_LABELS)}/{LIVE_CHECK_ONE_SCREEN_MAX_ROWS}"
            )
        return (
            False,
            f"acceptance coverage drift: {'; '.join(problems)} — repair live_check coverage before acceptance run",
        )

    covered_sections = len(ACCEPTANCE_HARNESS_SECTION_ROWS)
    covered_rows = len({row for rows in ACCEPTANCE_HARNESS_SECTION_ROWS.values() for row in rows})
    return (
        True,
        (
            f"acceptance coverage ready: {covered_sections}/6 DoD section(s) mapped across {covered_rows} diagnostic row(s) "
            f"with {mapped_item_count}/{plan_item_count} DoD checklist item(s) pinned to existing row evidence; "
            "for conversation/research, channels, personal integrations, voice, daily value, and reliability; "
            f"one-screen table budget {len(LIVE_CHECK_TABLE_LABELS)}/{LIVE_CHECK_ONE_SCREEN_MAX_ROWS} row(s); "
            "channel coverage includes channel health, send-resolution preflight, call-target readiness, "
            "and Telegram-owner readiness; "
            "personal integrations include direct Contacts lookup coverage; "
            "daily value includes scheduled-job freshness, 7-day proof, and phone approval-button flow; "
            "reliability includes acceptance coverage, open-gap visibility, next-proof guidance, "
            "aggregate smoke proof, operator-real workflow eval proof, error-guidance recovery checks, "
            "guardrail-plane shortcuts, proof-ledger shortcuts, completion-claim gating, and learning/recovery loop checks; "
            "read-only coverage map only, not a live proof for operator-gated sends/calls/approvals/microphone/reboot"
        ),
    )


def _acceptance_checkbox_counts(plan_text: str) -> tuple[dict[str, dict[str, int]], list[str]]:
    sections = {section: {"open": 0, "done": 0} for section in ACCEPTANCE_HARNESS_SECTION_ROWS}
    present_sections = {
        canonical
        for line in plan_text.splitlines()
        if line.startswith("### ")
        for canonical in (_canonical_acceptance_section(line[4:]),)
        if canonical is not None
    }
    missing_sections = [section for section in sections if section not in present_sections]
    current_section: str | None = None
    for line in plan_text.splitlines():
        if line.startswith("### "):
            current_section = _canonical_acceptance_section(line[4:])
            continue
        if current_section is None:
            continue
        match = ACCEPTANCE_CHECKBOX_RE.match(line)
        if not match:
            continue
        if match.group("state").lower() == "x":
            sections[current_section]["done"] += 1
        else:
            sections[current_section]["open"] += 1
    return sections, missing_sections


def _acceptance_offline_engineering_counts(plan_text: str) -> dict[str, int]:
    """Count open checklist items that require offline engineering, not a live proof."""

    counts: dict[str, int] = {}
    current_section: str | None = None
    for line in plan_text.splitlines():
        if line.startswith("### "):
            current_section = _canonical_acceptance_section(line[4:])
            continue
        if current_section is None:
            continue
        match = ACCEPTANCE_CHECKBOX_RE.match(line)
        if match is None or match.group("state").lower() == "x":
            continue
        item_text = line[match.end() :]
        for (section, anchor), label in ACCEPTANCE_OFFLINE_ENGINEERING_ITEMS.items():
            if current_section == section and item_text.startswith(anchor):
                counts[label] = counts.get(label, 0) + 1
                break
    return counts


def _check_acceptance_open_items(root: Path | None = None) -> tuple[bool | None, str]:
    """Read-only: how much of the August acceptance checklist is still open?"""
    plan_path = (root or REPO_ROOT) / "FINISH_PLAN_AUGUST.md"
    try:
        plan_text = plan_path.read_text(encoding="utf-8")
    except Exception as e:
        return False, f"acceptance gaps plan unreadable: {_safe_exception(e)}"

    counts, missing_sections = _acceptance_checkbox_counts(plan_text)
    if missing_sections:
        return (
            False,
            (
                f"acceptance gaps unavailable: missing DoD section(s): {', '.join(missing_sections[:3])}; "
                "repair finish-plan headings before acceptance run"
            ),
        )

    total_open = sum(count["open"] for count in counts.values())
    total_done = sum(count["done"] for count in counts.values())
    total_items = total_open + total_done
    if total_items == 0:
        return False, "acceptance gaps unavailable: no DoD checkbox items found; repair finish-plan checklist before acceptance run"

    open_sections = [
        f"{section} {count['open']}"
        for section, count in counts.items()
        if count["open"] > 0
    ]
    offline_engineering = _acceptance_offline_engineering_counts(plan_text)
    offline_engineering_total = sum(offline_engineering.values())
    operator_present_total = max(0, total_open - offline_engineering_total)
    open_work_types: list[str] = []
    if operator_present_total:
        open_work_types.append(
            f"operator-present live proofs still required ({operator_present_total})"
        )
    if offline_engineering_total:
        offline_labels = ", ".join(
            f"{label} {count}" for label, count in sorted(offline_engineering.items())
        )
        open_work_types.append(
            f"offline engineering debt remains ({offline_engineering_total}): {offline_labels}"
        )
    boundary = "read-only count only, no live sends/calls/approvals/microphone/reboot"
    if total_open:
        return (
            False,
            (
                f"acceptance gaps remain: {total_open} open / {total_done} done DoD checkbox(es); "
                f"open by section: {', '.join(open_sections)}; {'; '.join(open_work_types)}; "
                f"{ACCEPTANCE_GAPS_FOLLOWUP}; {boundary}"
            ),
        )

    return (
        True,
        (
            f"acceptance gaps clear: 0 open / {total_done} done DoD checkbox(es); "
            f"all {len(counts)} sections counted; run the final acceptance review with operator present; {boundary}"
        ),
    )


def _check_acceptance_next_action(root: Path | None = None) -> tuple[bool | None, str]:
    """Read-only: which acceptance lane should the operator prove next?

    This intentionally reports only a section-level lane and a safe evidence
    format. It never echoes checklist item text, recipients, prompts, message
    bodies, paths, or proof payloads.
    """
    plan_path = (root or REPO_ROOT) / "FINISH_PLAN_AUGUST.md"
    try:
        plan_text = plan_path.read_text(encoding="utf-8")
    except Exception as e:
        return False, f"acceptance next plan unreadable: {_safe_exception(e)}"

    counts, missing_sections = _acceptance_checkbox_counts(plan_text)
    if missing_sections:
        return (
            False,
            (
                f"acceptance next unavailable: missing DoD section(s): {', '.join(missing_sections[:3])}; "
                "repair finish-plan headings before acceptance run"
            ),
        )

    total_items = sum(count["open"] + count["done"] for count in counts.values())
    if total_items == 0:
        return False, "acceptance next unavailable: no DoD checkbox items found; repair finish-plan checklist before acceptance run"

    for section in ACCEPTANCE_NEXT_SECTION_ORDER:
        if counts.get(section, {}).get("open", 0) > 0:
            guidance = ACCEPTANCE_NEXT_GUIDANCE.get(section)
            if not guidance:
                return False, f"acceptance next unavailable: no guidance for {section}; repair live_check guidance"
            return (
                None,
                (
                    f"{guidance}; {counts[section]['open']} open item(s) in this lane"
                    f"{ACCEPTANCE_NEXT_REPORT_FORMAT}; "
                    "read-only guidance only, does not send/call/approve or replace operator-present proof"
                ),
            )

    total_done = sum(count["done"] for count in counts.values())
    return (
        True,
        (
            f"acceptance next clear: no open DoD checkbox(es), {total_done} done; "
            "run the final acceptance review with operator present; read-only guidance only"
        ),
    )


def _check_error_guidance_readiness(root: Path | None = None) -> tuple[bool | None, str]:
    """Read-only: do common recovery messages still name the next fix?

    This is not a comprehensive proof for every connector. It pins the
    high-signal recovery contracts that have already regressed or matter for
    the operator's acceptance run, without touching accounts, networks, models, audio,
    approvals, or channel sends.
    """
    problems: list[str] = []
    checked = 0

    try:
        import imaplib

        from jarvis_v2.agent.chat import ChatBrain
        from jarvis_v2.tools.apple_reminders import _apple_reminders_recovery_message
        from jarvis_v2.tools.audit import (
            _runtime_trace_missing_metadata_guidance,
            make_audit_tools,
        )
        from jarvis_v2.tools.browser import _browser_history_warning, _browser_state_error
        from jarvis_v2.tools.calendar_connector import _calendar_error
        from jarvis_v2.tools.call_connector import TelegramOwnerSendError, _telegram_error_stage
        from jarvis_v2.tools._http import HttpError, friendly_http_error
        from jarvis_v2.agent.model_planner import _model_exception_recovery_hint
        from jarvis_v2.tools.air_connector import _air_recovery_message
        from jarvis_v2.tools.compose_connector import _compose_model_recovery_hint
        from jarvis_v2.tools.currency_connector import _currency_recovery_message
        from jarvis_v2.tools.dictionary_connector import _dictionary_recovery_message
        from jarvis_v2.tools.decisions import _missing_decision_result
        from jarvis_v2.tools.email_connector import _credentials_error, _gmail_error
        from jarvis_v2.tools.files import _missing_text_file_result
        from jarvis_v2.tools.goals import _missing_goal_result, _missing_goal_step_result
        from jarvis_v2.tools.brief_tools import _daily_brief_recovery_line
        from jarvis_v2.tools.fun_connector import _joke_error_message
        from jarvis_v2.tools.history_connector import _history_recovery_message
        from jarvis_v2.tools.holidays_connector import _holidays_recovery_message
        from jarvis_v2.tools.imessage_connector import _imessage_error
        from jarvis_v2.tools.instagram_connector import _instagram_error
        from jarvis_v2.tools.kakao_connector import _kakao_error
        from jarvis_v2.tools.markets_connector import _market_error
        from jarvis_v2.tools.memory_curator import _missing_memory_result
        from jarvis_v2.tools.model_status import (
            _ollama_http_api_status,
            _ollama_unreachable_guidance,
        )
        from jarvis_v2.tools.news_connector import _news_recovery_message
        from jarvis_v2.tools.notes import _missing_note_result
        from jarvis_v2.tools.ocr import _ocr_read_error_message, _ocr_unavailable_message
        from jarvis_v2.tools.preferences import (
            _missing_preference_status_result,
            make_preference_tools,
        )
        from jarvis_v2.tools.reminder_tools import (
            _reminder_missing_owner_guidance,
            _reminder_storage_recovery_guidance,
            _reminder_time_parse_guidance,
        )
        from jarvis_v2.tools.research_connector import _search_recovery_message
        from jarvis_v2.tools.registry import _timezone_database_recovery_guidance
        from jarvis_v2.tools.startup_recovery import (
            _startup_recovery_audit_storage_guidance,
        )
        from jarvis_v2.tools.subagents import make_subagent_fleet_status_tool
        from jarvis_v2.tools.sun_connector import _sun_recovery_message
        from jarvis_v2.tools.tasks import (
            _missing_task_result,
            _note_task_not_found_result,
        )
        from jarvis_v2.tools.translate_connector import _translate_recovery_message
        from jarvis_v2.tools.voice import _voice_error
        from jarvis_v2.tools.weather_connector import _weather_recovery_message
        from jarvis_v2.tools.wikipedia_connector import _wiki_recovery_message
        from jarvis_v2.tools.writer_connector import _write_failure_message
        from jarvis_v2.automations.telegram_control import (
            _telegram_image_recovery_guidance,
            _telegram_voice_recovery_guidance,
            _telegram_voice_setup_guidance,
        )
        from jarvis_v2.automations.scheduler import _scheduler_recovery_guidance
    except Exception as e:
        return False, f"error guidance imports failed: {_safe_exception(e)}"

    class _MissingAuditStore:
        @staticmethod
        def recent_tool_runs(_limit: int):
            return [{"id": 1}]

        @staticmethod
        def get_tool_run(_run_id: int):
            return None

    class _MissingPreferenceStore:
        @staticmethod
        def get_preference(_key: str, *, category: str | None = None):
            return None

    audit_recovery = make_audit_tools(_MissingAuditStore())[4]({"run_id": 987654})
    preference_read = make_preference_tools(_MissingPreferenceStore(), object())[2](
        {"key": "missing-live-check-preference", "category": "general"}
    )
    chat_fallback = ChatBrain(
        model="jarvis-live-check-smoke-model",
        store=object(),
        provider="ollama",
    )._fallback_response(
        "hello",
        profile_context="",
        preference_context="",
        memory_context="",
        skill_context="",
        exc=None,
    )

    guidance_cases: tuple[tuple[str, object, tuple[str, ...], tuple[str, ...]], ...] = (
        (
            "Gmail credential setup guidance",
            "\n".join(
                (
                    _credentials_error({"reason": "missing_credentials"}),
                    _credentials_error({"reason": "invalid_credentials"}),
                    _gmail_error("read", imaplib.IMAP4.error("AUTHENTICATIONFAILED")),
                )
            ),
            (
                "GMAIL_ADDRESS and GMAIL_APP_PASSWORD must be set in .env",
                "fresh Gmail App Password",
                "myaccount.google.com/apppasswords",
                "values hidden",
            ),
            ERROR_GUIDANCE_FORBIDDEN_FRAGMENTS,
        ),
        (
            "local reminder setup guidance",
            "\n".join(
                (
                    _reminder_missing_owner_guidance(),
                    _reminder_storage_recovery_guidance(),
                    _reminder_time_parse_guidance(),
                )
            ),
            (
                "Telegram reminders need JARVIS_OWNER_TELEGRAM set",
                "Jarvis reminders file path is writable",
                "Try 'in 20 minutes'",
            ),
            ERROR_GUIDANCE_FORBIDDEN_FRAGMENTS,
        ),
        (
            "Ollama chat fallback guidance",
            chat_fallback,
            (
                "Start Ollama",
                "run `model status` in Jarvis",
                "ollama pull",
                "Diagnostic: model_unavailable",
            ),
            ERROR_GUIDANCE_FORBIDDEN_FRAGMENTS,
        ),
        (
            "Browser state guidance",
            _tool_result_guidance(
                _browser_state_error(
                    "save_page_note",
                    "access browser history or the Jarvis note vault",
                    "save latest page note",
                    needs_vault=True,
                )
            ).output
            + "\n"
            + _browser_history_warning("Page fetch", RuntimeError("SHOULD NOT APPEAR")),
            (
                "History warning:",
                "`JARVIS_DATA_DIR` and `JARVIS_DB_PATH`",
                "`JARVIS_OBSIDIAN_VAULT`",
                "Run `setup check`",
            ),
            ERROR_GUIDANCE_FORBIDDEN_FRAGMENTS + ("SHOULD NOT APPEAR",),
        ),
        (
            "Worker fleet recovery guidance",
            _tool_result_guidance(make_subagent_fleet_status_tool(None)({})),
            (
                "restart Jarvis with its normal launcher",
                "`subagent fleet status`",
                '"authorizes_worker_dispatch": false',
                '"authorizes_restart": false',
            ),
            ERROR_GUIDANCE_FORBIDDEN_FRAGMENTS,
        ),
        (
            "Audit lookup recovery guidance",
            _tool_result_guidance(audit_recovery),
            (
                "refresh valid IDs",
                '"retry_requires_audit_refresh": true',
                '"retry_requires_storage_repair": false',
                '"authorizes_retry": false',
            ),
            ERROR_GUIDANCE_FORBIDDEN_FRAGMENTS,
        ),
        (
            "timezone database recovery guidance",
            _timezone_database_recovery_guidance(),
            ("setup check", "tzdata", "retry", "UTC"),
            ERROR_GUIDANCE_FORBIDDEN_FRAGMENTS,
        ),
        (
            "startup recovery audit storage guidance",
            _startup_recovery_audit_storage_guidance(),
            ("storage status", "setup check", "repair", "startup recovery report"),
            ERROR_GUIDANCE_FORBIDDEN_FRAGMENTS,
        ),
        (
            "runtime trace missing-metadata guidance",
            _runtime_trace_missing_metadata_guidance(),
            ("Send a Jarvis command first", "runtime trace receipt", "without an ID"),
            ERROR_GUIDANCE_FORBIDDEN_FRAGMENTS,
        ),
        (
            "Local file not-found guidance",
            _tool_result_guidance(
                _missing_text_file_result(
                    Path("missing-live-check-file.txt"),
                    max_chars=5000,
                )
            ),
            (
                '"retry_requires_path_refresh": true',
                '"retry_requires_fresh_approval": true',
                "through the normal approval flow",
            ),
            ERROR_GUIDANCE_FORBIDDEN_FRAGMENTS,
        ),
        (
            "Jarvis note not-found guidance",
            _tool_result_guidance(_missing_note_result("Missing/Live Check.md")),
            (
                '"retry_requires_note_refresh": true',
                "Run `list jarvis notes`",
                '"authorizes_retry": false',
            ),
            ERROR_GUIDANCE_FORBIDDEN_FRAGMENTS,
        ),
        (
            "Note-to-task not-found guidance",
            _tool_result_guidance(
                _note_task_not_found_result(
                    tool_name="preview_tasks_from_note",
                    raw_path="Missing/Tasks.md",
                ),
                _note_task_not_found_result(
                    tool_name="import_tasks_from_note",
                    raw_path="Missing/Tasks.md",
                ),
                per_result_required=(
                    (
                        ("preview tasks from note",),
                        (
                            '"retry_requires_note_refresh": true',
                            '"authorizes_retry": false',
                        ),
                    ),
                    (
                        ("import tasks from note",),
                        (
                            '"retry_requires_note_refresh": true',
                            '"authorizes_retry": false',
                        ),
                    ),
                ),
            ),
            (
                '"retry_requires_note_refresh": true',
                "preview tasks from note",
                "import tasks from note",
            ),
            ERROR_GUIDANCE_FORBIDDEN_FRAGMENTS,
        ),
        (
            "Task-id not-found guidance",
            _tool_result_guidance(
                _missing_task_result(
                    tool_name="inspect_task",
                    task_id=987654,
                    mutation="task_read",
                    retry_command="show task <correct task id>",
                )
            ),
            (
                '"retry_requires_task_refresh": true',
                "show task <correct task id>",
                '"authorizes_retry": false',
            ),
            ERROR_GUIDANCE_FORBIDDEN_FRAGMENTS,
        ),
        (
            "Goal-id not-found guidance",
            _tool_result_guidance(
                _missing_goal_result(
                    tool_name="goal_status",
                    goal_id=987654,
                    mutation="goal_read",
                    retry_command="goal <correct goal id> status",
                ),
                _missing_goal_step_result(step_id=987654),
                per_result_required=(
                    (
                        ("goal <correct goal id> status",),
                        (
                            '"retry_requires_goal_refresh": true',
                            '"authorizes_retry": false',
                        ),
                    ),
                    (
                        ("complete goal step <correct step id>",),
                        (
                            '"retry_requires_goal_step_refresh": true',
                            '"authorizes_retry": false',
                        ),
                    ),
                ),
            ),
            (
                '"retry_requires_goal_refresh": true',
                '"retry_requires_goal_step_refresh": true',
                "goal <correct goal id> status",
                '"authorizes_retry": false',
            ),
            ERROR_GUIDANCE_FORBIDDEN_FRAGMENTS,
        ),
        (
            "Memory-id not-found guidance",
            _tool_result_guidance(
                _missing_memory_result(
                    tool_name="get_memory",
                    memory_id=987654,
                    mutation="memory_read",
                    requires_approval=False,
                ),
                _missing_memory_result(
                    tool_name="delete_memory",
                    memory_id=987654,
                    mutation="memory_delete",
                    requires_approval=True,
                    retry_command="delete memory <correct memory id>",
                ),
                per_result_required=(
                    (
                        ("show memory <correct memory id>",),
                        (
                            '"retry_requires_memory_refresh": true',
                            '"authorizes_retry": false',
                        ),
                    ),
                    (
                        ("delete memory <correct memory id>", "does not reuse or grant approval"),
                        (
                            '"retry_requires_memory_refresh": true',
                            '"retry_requires_fresh_approval": true',
                            '"authorizes_retry": false',
                        ),
                    ),
                ),
            ),
            (
                '"retry_requires_memory_refresh": true',
                '"retry_requires_fresh_approval": true',
                "show memory <correct memory id>",
                "does not reuse or grant approval",
                '"authorizes_retry": false',
            ),
            ERROR_GUIDANCE_FORBIDDEN_FRAGMENTS,
        ),
        (
            "Decision-id not-found guidance",
            _tool_result_guidance(
                _missing_decision_result(
                    tool_name="get_decision",
                    decision_id=987654,
                    mutation="decision_read",
                    read_only=True,
                    retry_command="show decision <correct decision id>",
                )
            ),
            (
                '"retry_requires_decision_refresh": true',
                "show decision <correct decision id>",
                '"authorizes_retry": false',
            ),
            ERROR_GUIDANCE_FORBIDDEN_FRAGMENTS,
        ),
        (
            "Preference not-found guidance",
            _tool_result_guidance(
                preference_read,
                _missing_preference_status_result(preference_id=987654, status="retired"),
                per_result_required=(
                    (
                        ("set preference <key> to <value> category <category>",),
                        (
                            '"retry_requires_preference_refresh": true',
                            '"authorizes_retry": false',
                        ),
                    ),
                    (
                        ("show preference <correct preference key>",),
                        (
                            '"retry_requires_preference_refresh": true',
                            '"authorizes_retry": false',
                        ),
                    ),
                ),
            ),
            (
                '"retry_requires_preference_refresh": true',
                "show preference <correct preference key>",
                "set preference <key> to <value> category <category>",
                '"authorizes_retry": false',
            ),
            ERROR_GUIDANCE_FORBIDDEN_FRAGMENTS,
        ),
        (
            "telegram voice setup guidance",
            _telegram_voice_setup_guidance(),
            (
                "Voice transcription isn't set up",
                "Install the `whisper` CLI",
                "JARVIS_VOICE_WHISPER_MODEL_PATH",
                "try again",
            ),
            ERROR_GUIDANCE_FORBIDDEN_FRAGMENTS,
        ),
        (
            "Telegram voice media recovery guidance",
            "\n".join(
                _telegram_voice_recovery_guidance(stage)
                for stage in ("invalid_media", "download", "transcribe", "empty")
            ),
            (
                "under 20MB",
                "Check Telegram network access",
                "`voice setup check`",
                "type the command instead",
            ),
            ERROR_GUIDANCE_FORBIDDEN_FRAGMENTS,
        ),
        (
            "Telegram image OCR media recovery guidance",
            "\n".join(
                _telegram_image_recovery_guidance(stage)
                for stage in ("invalid_media", "unavailable", "download", "read", "empty")
            ),
            (
                "PNG/JPG",
                "`setup check`",
                "`swift --version`",
                "Check Telegram network access",
                "type the text instead",
            ),
            ERROR_GUIDANCE_FORBIDDEN_FRAGMENTS,
        ),
        (
            "Apple Reminders permission guidance",
            _apple_reminders_recovery_message(),
            (
                "I couldn't verify your Apple Reminders right now",
                "Check Reminders access",
                "macOS System Settings",
                "then retry",
            ),
            ERROR_GUIDANCE_FORBIDDEN_FRAGMENTS,
        ),
        (
            "voice recovery guidance",
            "\n".join(
                (
                    _voice_error(
                        "speak text",
                        RuntimeError("/\x55sers/operator/private/speech SHOULD NOT APPEAR"),
                    ),
                    _voice_error(
                        "transcribe approved audio",
                        RuntimeError("/\x55sers/operator/private/audio SHOULD NOT APPEAR"),
                    ),
                )
            ),
            (
                "Check System Settings > Accessibility > Spoken Content",
                "System Settings > Sound > Output",
                "Check audio-file permissions and ASR dependencies",
                "voice setup check",
                "then retry",
            ),
            ERROR_GUIDANCE_FORBIDDEN_FRAGMENTS + ("SHOULD NOT APPEAR", "private/speech", "private/audio"),
        ),
        (
            "Ollama status guidance",
            "\n".join(
                (
                    _ollama_http_api_status(False),
                    _ollama_unreachable_guidance(
                        "/\x55sers/operator/private/model SHOULD NOT APPEAR"
                    ),
                )
            ),
            (
                "Ollama HTTP API: needs attention",
                "Start Ollama, then retry `model routing status`; run `ollama pull ",
            ),
            ERROR_GUIDANCE_FORBIDDEN_FRAGMENTS
            + (
                "SHOULD NOT APPEAR",
                "private/model",
                "install the Ollama Python package",
            ),
        ),
        (
            # Calls the real function and checks its RETURN VALUE rather than
            # scanning model_planner.py's source text: the diagnostic suffix is
            # built via `f"Diagnostic: {MODEL_EXCEPTION_RECOVERY_DIAGNOSTIC}."`,
            # so the literal substring never appears contiguous in the source --
            # a static text scan there always false-fails regardless of actual
            # (correct) runtime behavior. This case previously lived in
            # SOURCE_ERROR_GUIDANCE_REQUIRED_TEXT and broke exactly that way.
            "Ollama planner fallback guidance",
            _model_exception_recovery_hint("jarvis-live-check-smoke-model"),
            ("model routing status", "start Ollama", "ollama pull", "Diagnostic: planner_model_unavailable"),
            ERROR_GUIDANCE_FORBIDDEN_FRAGMENTS,
        ),
        (
            "calendar dead-token guidance",
            _calendar_error(Exception("('invalid_grant: Bad Request', {'error': 'invalid_grant'})")),
            CALENDAR_READONLY_GUIDANCE_REQUIRED + ("Retrying will not help",),
            CALENDAR_READONLY_GUIDANCE_FORBIDDEN + ("try again in a moment",),
        ),
        (
            "calendar missing-setup guidance",
            _calendar_error(Exception("Google credentials not found at /\x55sers/secret/creds.json. Run setup first.")),
            ("not connected yet", "Google credentials", "run setup"),
            ERROR_GUIDANCE_FORBIDDEN_FRAGMENTS + ("creds.json",),
        ),
        (
            "calendar generic recovery guidance",
            _calendar_error(Exception("raw calendar backend exploded SHOULD NOT APPEAR")),
            (
                "network access to Google Calendar",
                "setup check",
                *CALENDAR_READONLY_GUIDANCE_REQUIRED,
                "retry",
            ),
            CALENDAR_READONLY_GUIDANCE_FORBIDDEN + ("SHOULD NOT APPEAR", "backend exploded"),
        ),
        (
            "HTTP not-found guidance",
            friendly_http_error(HttpError(404, "HTTP Error 404: Not Found"), subject="weather for 'badcity'", service="Weather"),
            ("couldn't find",),
            ERROR_GUIDANCE_FORBIDDEN_FRAGMENTS,
        ),
        (
            "HTTP service guidance",
            friendly_http_error(HttpError(503, "HTTP Error 503: Service Unavailable"), service="Weather"),
            ("Weather is having trouble", "try again in a moment"),
            ERROR_GUIDANCE_FORBIDDEN_FRAGMENTS,
        ),
        (
            "HTTP timeout guidance",
            friendly_http_error(TimeoutError("socket timeout SHOULD NOT APPEAR"), service="News"),
            ("timed out", "try again"),
            ERROR_GUIDANCE_FORBIDDEN_FRAGMENTS + ("SHOULD NOT APPEAR", "socket timeout"),
        ),
        (
            "daily brief degraded-section guidance",
            _daily_brief_recovery_line(["weather", "calendar", "reminders", "news"]),
            (
                "weather: check network access to wttr.in",
                "calendar: check Google Calendar network/auth",
                *CALENDAR_READONLY_GUIDANCE_REQUIRED,
                "reminders: check the local Jarvis reminders file path is writable",
                "news: check network access to Google News",
                "setup check",
                "daily briefing",
            ),
            CALENDAR_READONLY_GUIDANCE_FORBIDDEN,
        ),
        (
            "weather connector recovery guidance",
            _weather_recovery_message("Weather data was unavailable."),
            ("network access to wttr.in", "setup check", "retry"),
            ERROR_GUIDANCE_FORBIDDEN_FRAGMENTS,
        ),
        (
            "news connector recovery guidance",
            _news_recovery_message("News headlines were unavailable."),
            ("network access to Google News", "setup check", "retry"),
            ERROR_GUIDANCE_FORBIDDEN_FRAGMENTS,
        ),
        (
            "currency connector recovery guidance",
            _currency_recovery_message("Currency rates were unavailable."),
            ("network access to Frankfurter", "setup check", "retry"),
            ERROR_GUIDANCE_FORBIDDEN_FRAGMENTS,
        ),
        (
            "translation connector recovery guidance",
            _translate_recovery_message("Translation was unavailable."),
            ("network access to MyMemory", "setup check", "retry"),
            ERROR_GUIDANCE_FORBIDDEN_FRAGMENTS,
        ),
        (
            "sunrise connector recovery guidance",
            _sun_recovery_message("Sunrise/sunset data was unavailable."),
            ("network access to sunrise-sunset.org", "setup check", "retry"),
            ERROR_GUIDANCE_FORBIDDEN_FRAGMENTS,
        ),
        (
            "history connector recovery guidance",
            _history_recovery_message("History data was unavailable."),
            ("network access to Wikipedia on-this-day", "setup check", "retry"),
            ERROR_GUIDANCE_FORBIDDEN_FRAGMENTS,
        ),
        (
            "Wikipedia connector recovery guidance",
            _wiki_recovery_message("Wikipedia summary was unavailable."),
            ("network access to Wikipedia", "setup check", "retry"),
            ERROR_GUIDANCE_FORBIDDEN_FRAGMENTS,
        ),
        (
            "research connector recovery guidance",
            _search_recovery_message("Search results were unavailable."),
            ("network access to DuckDuckGo Lite", "setup check", "retry"),
            ERROR_GUIDANCE_FORBIDDEN_FRAGMENTS,
        ),
        (
            "dictionary connector recovery guidance",
            _dictionary_recovery_message("Dictionary data was unavailable."),
            ("network access to dictionaryapi.dev", "setup check", "retry"),
            ERROR_GUIDANCE_FORBIDDEN_FRAGMENTS,
        ),
        (
            "air-quality connector recovery guidance",
            _air_recovery_message("Air-quality data was unavailable."),
            ("network access to Open-Meteo air quality", "setup check", "retry"),
            ERROR_GUIDANCE_FORBIDDEN_FRAGMENTS,
        ),
        (
            "holidays connector recovery guidance",
            _holidays_recovery_message("Holiday data was unavailable."),
            ("network access to Nager.Date holidays", "setup check", "retry"),
            ERROR_GUIDANCE_FORBIDDEN_FRAGMENTS,
        ),
        (
            "markets connector recovery guidance",
            _market_error(
                RuntimeError("/\x55sers/operator/private/market backend SHOULD NOT APPEAR"),
                subject="crypto prices",
                service="CoinGecko",
            ),
            ("network access to CoinGecko", "setup check", "retry"),
            ERROR_GUIDANCE_FORBIDDEN_FRAGMENTS + ("SHOULD NOT APPEAR", "market backend"),
        ),
        (
            "joke connector recovery guidance",
            _joke_error_message(RuntimeError("/\x55sers/operator/private/joke backend SHOULD NOT APPEAR")),
            ("network access to icanhazdadjoke.com", "setup check", "tell me a joke"),
            ERROR_GUIDANCE_FORBIDDEN_FRAGMENTS + ("SHOULD NOT APPEAR", "joke backend"),
        ),
        (
            "compose-and-write model guidance",
            _compose_model_recovery_hint(SimpleNamespace(chat_model="/\x55sers/operator/private/model SHOULD NOT APPEAR")),
            ("model routing status", "start Ollama", "ollama pull", "compose_model_unavailable"),
            ERROR_GUIDANCE_FORBIDDEN_FRAGMENTS + ("SHOULD NOT APPEAR", "private/model"),
        ),
        (
            "writer type recovery guidance",
            _write_failure_message("human"),
            ("target app focused", "Accessibility", "System Settings", "setup check", "after approval"),
            ERROR_GUIDANCE_FORBIDDEN_FRAGMENTS,
        ),
        (
            "writer paste recovery guidance",
            _write_failure_message("paste"),
            ("target app focused", "Accessibility", "System Settings", "setup check", "after approval"),
            ERROR_GUIDANCE_FORBIDDEN_FRAGMENTS,
        ),
        (
            "OCR unavailable guidance",
            _ocr_unavailable_message(),
            ("On-device OCR", "setup check", "swift --version", "Apple Command Line Tools", "ocr image: <path>"),
            ERROR_GUIDANCE_FORBIDDEN_FRAGMENTS,
        ),
        (
            "OCR read recovery guidance",
            _ocr_read_error_message(),
            ("opens in Preview", "PNG or JPG", "setup check", "ocr image: <path>"),
            ERROR_GUIDANCE_FORBIDDEN_FRAGMENTS,
        ),
        (
            "Telegram owner-channel recovery guidance",
            str(
                TelegramOwnerSendError(
                    "owner_not_configured",
                    "Configure the owner Telegram channel, then retry.",
                )
            ),
            ("owner_not_configured", "Configure the owner Telegram channel", "retry"),
            ERROR_GUIDANCE_FORBIDDEN_FRAGMENTS,
        ),
        (
            "Telegram Web permission guidance",
            " ".join(
                _telegram_error_stage(
                    "System Events got an error: osascript is not allowed assistive access"
                )
            ),
            ("macos_accessibility_permission", "Grant Accessibility permission", "then retry"),
            ERROR_GUIDANCE_FORBIDDEN_FRAGMENTS,
        ),
        (
            "KakaoTalk send recovery guidance",
            _kakao_error(),
            ("KakaoTalk is installed", "logged in", "Accessibility permission"),
            ERROR_GUIDANCE_FORBIDDEN_FRAGMENTS,
        ),
        (
            "Instagram DM recovery guidance",
            _instagram_error(),
            ("Chrome is logged into Instagram", "Accessibility permission"),
            ERROR_GUIDANCE_FORBIDDEN_FRAGMENTS,
        ),
        (
            "iMessage send recovery guidance",
            _imessage_error("send"),
            ("Check Messages setup", "try again"),
            ERROR_GUIDANCE_FORBIDDEN_FRAGMENTS,
        ),
        (
            "scheduler execution and delivery recovery guidance",
            "\n".join(
                (
                    _scheduler_recovery_guidance(),
                    _scheduler_recovery_guidance(
                        automatic_retry=False,
                        delivery_receipt_may_be_unknown=True,
                    ),
                )
            ),
            (
                "list scheduled jobs",
                "setup check",
                "storage, model, or connector issue",
                "automatic retry",
                "durable delivery receipt state",
                "do not manually rerun or resend",
                "outcome-unknown delivery",
            ),
            ERROR_GUIDANCE_FORBIDDEN_FRAGMENTS,
        ),
    )
    inventory_problem = _error_guidance_inventory_problem(
        tuple(label for label, _message, _required, _forbidden in guidance_cases)
    )
    if inventory_problem:
        problems.append(inventory_problem)
    for label, message, required, forbidden in guidance_cases:
        checked += 1
        problem = _guidance_problem(label, message, required=required, forbidden=forbidden)
        if problem:
            problems.append(problem)

    if problems:
        return (
            False,
            (
                f"error guidance drift {len(problems)} contract(s): {', '.join(problems[:4])} — "
                "repair recovery wording before acceptance run"
            ),
        )

    return (
        True,
        (
            f"error guidance ready: {checked} recovery contract(s); Calendar least-privilege root-launcher reauth/missing setup/generic recovery, "
            "HTTP not-found/service/timeout, Gmail, reminders, voice, Ollama chat/planner/status, and Telegram voice setup guidance plus media/image OCR recovery name a next fix; "
            "timezone data, startup recovery audit storage, and missing runtime-trace metadata failures name concrete diagnostic and retry steps; "
            "common info connectors name service-specific fixes for weather, news, currency, translation, sun, history, Wikipedia, research, dictionary, air quality, holidays, markets, and jokes; "
            "daily brief degraded-section guidance names weather/calendar/reminders/news fixes; "
            "compose/write/OCR guidance also names model routing, Accessibility, and local OCR fixes; "
            "Telegram owner/Web, KakaoTalk, Instagram, and iMessage send failures name channel-specific fixes; "
            "scheduler execution, recovery, and outcome-unknown delivery failures name inspection and safe no-resend steps; "
            "no account access, web fetch, model call, transcription, approval, typing, pasting, OCR execution, or send run"
        ),
    )


def _check_image_ocr() -> tuple[bool | None, str]:
    """Read-only: is on-device image OCR available (for photo/document intake)?"""
    try:
        from jarvis_v2.tools import ocr

        if not ocr.ocr_available():
            return False, "no on-device OCR — needs macOS Swift/Vision (swiftc/swift)"
    except Exception as e:
        return False, f"EXCEPTION: {_safe_exception(e)}"
    return True, "ready via macOS Vision (Swift)"


def _check_daily_brief(config) -> tuple[bool | None, str]:
    try:
        from jarvis_v2.tools.brief_tools import make_brief_tools

        tool = {t.name: t for t in make_brief_tools(config)}["daily_briefing"]
        result = tool.handler({})
    except Exception as e:
        return False, f"EXCEPTION: {_safe_exception(e)}"
    output = result.output
    if not isinstance(output, str):
        return False, "daily_briefing returned non-text content"
    if not result.ok or not output.strip():
        return False, "daily_briefing returned no content"
    line_count = len(output.splitlines())
    metadata = getattr(result, "metadata", {})
    section_detail = ""
    available_count = _safe_count(_metadata_value(metadata, "available_count"))
    unavailable_count = _safe_count(_metadata_value(metadata, "unavailable_count"))
    recovery_hint_count = _safe_count(_metadata_value(metadata, "daily_brief_recovery_hint_count"))
    if recovery_hint_count == "unknown":
        recovery_hint_count = _safe_count(_metadata_value(metadata, "recovery_hint_count"))
    if recovery_hint_count == "unknown":
        handoff = _metadata_value(metadata, "daily_brief_handoff", {})
        if isinstance(handoff, dict):
            recovery_hint_count = _safe_count(_metadata_value(handoff, "recovery_hint_count"))
    if available_count != "unknown" and unavailable_count != "unknown":
        section_detail = f"; {available_count} section(s) available, {unavailable_count} unavailable"
        unavailable_int = int(unavailable_count)
        if unavailable_int > 0:
            if recovery_hint_count == "unknown":
                return False, (
                    f"daily_briefing reported {unavailable_count} unavailable section(s) without recovery hint proof; "
                    "content suppressed"
                )
            if int(recovery_hint_count) < unavailable_int:
                return False, (
                    f"daily_briefing reported {unavailable_count} unavailable section(s) but only "
                    f"{recovery_hint_count} recovery hint(s); content suppressed"
                )
            section_detail += f"; {recovery_hint_count} recovery hint(s)"
    return True, f"composed {len(output)} chars across {line_count} line(s){section_detail}; content suppressed"


def _check_research_readiness(config) -> tuple[bool | None, str]:
    """Read-only: are research/web lookup routes and tool risk gates intact?"""
    try:
        from jarvis_v2.agent.planner import RuleBasedPlanner
        from jarvis_v2.memory.obsidian import ObsidianVault
        from jarvis_v2.memory.store import MemoryStore
        from jarvis_v2.tools.registry import build_core_registry

        registry = build_core_registry(
            MemoryStore(config.db_path),
            ObsidianVault(config.obsidian_vault, config.obsidian_root),
            config,
        )
    except Exception as e:
        return False, f"research registry unreadable: {_safe_exception(e)}"

    missing: list[str] = []
    wrong_risk: list[str] = []
    for tool_name, expected_risk in RESEARCH_TOOL_RISKS.items():
        try:
            tool = registry.get(tool_name)
            actual_risk = _plain_text(getattr(tool.risk, "name", tool.risk))
        except Exception:
            missing.append(tool_name)
            continue
        if actual_risk != expected_risk:
            wrong_risk.append(f"{tool_name}:{actual_risk or 'unknown'}")

    route_drift: list[str] = []
    try:
        planner = RuleBasedPlanner()
        for phrase, expected_tool, expected_query in RESEARCH_ROUTE_CASES:
            actions = planner.plan(phrase).actions
            action = actions[0] if len(actions) == 1 else None
            actual_tool = _plain_text(getattr(action, "tool_name", "")) if action is not None else ""
            actual_query = ""
            if action is not None:
                actual_query = _plain_text(getattr(action, "args", {}).get("query"))
            if actual_tool != expected_tool or actual_query != expected_query:
                route_drift.append(expected_tool)
    except Exception as e:
        return False, f"research planner unreadable: {_safe_exception(e)}"

    if missing or wrong_risk or route_drift:
        parts = []
        if missing:
            parts.append(f"missing {len(missing)} tool(s): {', '.join(sorted(missing)[:4])}")
        if wrong_risk:
            parts.append(f"wrong risk {len(wrong_risk)} tool(s): {', '.join(sorted(wrong_risk)[:4])}")
        if route_drift:
            drift_counts = {tool: route_drift.count(tool) for tool in sorted(set(route_drift))}
            drift_summary = ", ".join(f"{tool}={count}" for tool, count in drift_counts.items())
            parts.append(f"route drift {len(route_drift)} case(s): {drift_summary}")
        return False, "; ".join(parts) + " — fix research routing/tool gates before live sourced-answer proof"

    return (
        True,
        (
            f"tool gates ready: {len(RESEARCH_TOOL_RISKS)}/{len(RESEARCH_TOOL_RISKS)}; "
            f"routes ready: {len(RESEARCH_ROUTE_CASES)}/{len(RESEARCH_ROUTE_CASES)}; "
            "research/web lookup remain READ_ONLY/LOCAL_SAFE; no web fetch, model call, note write, or approval run"
        ),
    )


def _check_personal_route_readiness() -> tuple[bool | None, str]:
    """Read-only: do personal-integration commands reach the intended safe path?"""
    try:
        from jarvis_v2.agent.planner import RuleBasedPlanner

        planner = RuleBasedPlanner()
    except Exception as e:
        return False, f"personal route planner unreadable: {_safe_exception(e)}"

    route_drift: list[str] = []
    arg_drift: list[str] = []
    dispatch_cases = 0
    direct_cases = 0
    try:
        for phrase, expected_tool, expected_args, required_keys in PERSONAL_ROUTE_CASES:
            actions = planner.plan(phrase).actions
            action = actions[0] if len(actions) == 1 else None
            actual_tool = _plain_text(getattr(action, "tool_name", "")) if action is not None else ""
            args = getattr(action, "args", {}) if action is not None else {}
            args = args if isinstance(args, dict) else {}
            if actual_tool != expected_tool:
                route_drift.append(expected_tool)
                continue
            missing_or_changed = [
                key for key, expected_value in expected_args.items()
                if args.get(key) != expected_value
            ]
            missing_or_changed.extend(key for key in required_keys if key not in args or not args.get(key))
            if missing_or_changed:
                arg_drift.append(expected_tool)
            if expected_tool == "dispatch_decision_packet":
                dispatch_cases += 1
            else:
                direct_cases += 1
    except Exception as e:
        return False, f"personal route check failed closed: {_safe_exception(e)}"

    if route_drift or arg_drift:
        parts = []
        if route_drift:
            route_counts = {tool: route_drift.count(tool) for tool in sorted(set(route_drift))}
            parts.append(
                "route drift "
                + ", ".join(f"{tool}={count}" for tool, count in route_counts.items())
            )
        if arg_drift:
            arg_counts = {tool: arg_drift.count(tool) for tool in sorted(set(arg_drift))}
            parts.append(
                "arg drift "
                + ", ".join(f"{tool}={count}" for tool, count in arg_counts.items())
            )
        return False, "; ".join(parts) + " — fix personal command routing before live write proofs"

    return (
        True,
        (
            f"routes ready: {len(PERSONAL_ROUTE_CASES)}/{len(PERSONAL_ROUTE_CASES)}; "
            f"direct personal paths {direct_cases}; dispatch-held risky paths {dispatch_cases}; "
            "calendar/email/reminder aliases do not fall to chat; no handlers, approvals, account reads, or sends run"
        ),
    )


def _check_chat_path_health(config) -> tuple[bool | None, str]:
    """Read-only: recent ordinary chat should be model-backed and latency-instrumented."""
    try:
        from jarvis_v2.memory.store import MemoryStore

        store = MemoryStore(config.db_path)
        rows = store.recent_messages(limit=CHAT_PATH_RECENT_LIMIT, session_id=None)
    except Exception as e:
        return False, f"chat store unreadable: {_safe_exception(e)}"

    ordinary_responses: list[object] = []
    for row in rows or []:
        if _plain_text(_mapping_value(row, "role")).casefold() != "assistant":
            continue
        metadata = _message_metadata(row)
        if _metadata_value(metadata, "runtime_route") != "chat":
            continue
        response = _metadata_value(metadata, "chat_response", {})
        if not isinstance(response, dict):
            continue
        if _metadata_value(response, "runtime_route") == "command_suggestion":
            continue
        if _metadata_value(response, "pre_planner_command_suggestion") is True:
            continue
        ordinary_responses.append(response)

    if not ordinary_responses:
        return (
            None,
            (
                f"no stored ordinary chat turns in the last {CHAT_PATH_RECENT_LIMIT} messages — "
                "ask Jarvis a normal question, then rerun live_check"
            ),
        )

    source_counts: dict[str, int] = {}
    model_count = 0
    fallback_count = 0
    latencies: list[float] = []
    for response in ordinary_responses:
        source = _safe_chat_source(_metadata_value(response, "source", "unknown"))
        source_counts[source] = source_counts.get(source, 0) + 1
        if _metadata_value(response, "used_model") is True:
            model_count += 1
        if _metadata_value(response, "used_fallback") is True:
            fallback_count += 1
        latency = _chat_latency_ms(response)
        if latency is not None:
            latencies.append(latency)

    ordinary_count = len(ordinary_responses)
    source_bits = ", ".join(f"{source}={count}" for source, count in sorted(source_counts.items()))
    detail = (
        f"{ordinary_count} recent ordinary chat turn(s); model {model_count}; fallback {fallback_count}; "
        f"latency samples {len(latencies)}"
    )
    if latencies:
        detail += f"; p50/p95 {_percentile(latencies, 0.50)}ms/{_percentile(latencies, 0.95)}ms"
    detail += f"; sources {source_bits or 'none'}; metadata-only"

    if not latencies:
        return False, detail + " — run one normal chat turn after the latest build"
    if model_count == 0:
        return False, detail + " — no model-backed ordinary chat in recent stored turns"
    if fallback_count > 0 and fallback_count >= model_count:
        return False, detail + " — fallback count is too high; inspect chat_response_health"
    if ordinary_count < CHAT_PATH_MIN_READY_TURNS:
        return (
            None,
            detail + f" — collect {CHAT_PATH_MIN_READY_TURNS - ordinary_count} more normal chat turn(s)",
        )
    return True, detail


def _check_mixed_conversation_health(config) -> tuple[bool | None, str]:
    """Read-only: has a recent mixed session proven chat+research+calendar+tasks?"""
    try:
        from jarvis_v2.memory.store import MemoryStore

        store = MemoryStore(config.db_path)
        sessions = store.list_sessions(limit=MIXED_CONVERSATION_RECENT_SESSIONS)
    except Exception as e:
        return False, f"conversation store unreadable: {_safe_exception(e)}"

    if not sessions:
        return (
            None,
            "no stored sessions — run a mixed chat/research/calendar/tasks conversation, then rerun live_check",
        )

    best: dict[str, object] | None = None
    for session in sessions:
        session_id = _plain_text(_mapping_value(session, "session_id")).strip()
        if not session_id:
            continue
        try:
            rows = store.recent_messages(limit=MIXED_CONVERSATION_MESSAGE_LIMIT, session_id=session_id)
        except Exception:
            continue

        assistant_turns = 0
        lane_counts = {lane: 0 for lane in MIXED_CONVERSATION_REQUIRED_LANES}
        model_count = 0
        fallback_count = 0
        latencies: list[float] = []

        for row in rows or []:
            if _plain_text(_mapping_value(row, "role")).casefold() != "assistant":
                continue
            metadata = _message_metadata(row)
            trace = _runtime_trace(metadata)
            route = _metadata_value(metadata, "runtime_route") or _metadata_value(trace, "route")
            if not route:
                continue
            assistant_turns += 1

            response = _ordinary_chat_response(metadata)
            if response is not None:
                lane_counts["chat"] += 1
                if _metadata_value(response, "used_model") is True:
                    model_count += 1
                if _metadata_value(response, "used_fallback") is True:
                    fallback_count += 1
                latency = _chat_latency_ms(response)
                if latency is not None:
                    latencies.append(latency)

            for action in _planned_actions(trace):
                for lane in _mixed_action_lanes(action):
                    lane_counts[lane] += 1

        present_lanes = sum(1 for lane in MIXED_CONVERSATION_REQUIRED_LANES if lane_counts[lane] > 0)
        score = (present_lanes, assistant_turns, len(latencies), model_count)
        candidate = {
            "score": score,
            "assistant_turns": assistant_turns,
            "lane_counts": lane_counts,
            "model_count": model_count,
            "fallback_count": fallback_count,
            "latencies": latencies,
        }
        if best is None or score > best["score"]:
            best = candidate

    if best is None:
        return (
            None,
            "no readable recent session metadata — run a mixed conversation after this build, then rerun live_check",
        )

    lane_counts = best["lane_counts"] if isinstance(best.get("lane_counts"), dict) else {}
    assistant_turns = _safe_positive_int(best.get("assistant_turns")) or 0
    model_count = _safe_positive_int(best.get("model_count")) or 0
    fallback_count = _safe_positive_int(best.get("fallback_count")) or 0
    raw_latencies = best.get("latencies") if isinstance(best.get("latencies"), list) else []
    latencies = [
        float(value)
        for value in raw_latencies
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))
    ]
    missing = [lane for lane in MIXED_CONVERSATION_REQUIRED_LANES if _safe_positive_int(lane_counts.get(lane)) is None]
    lane_bits = ", ".join(
        f"{lane}={_safe_count(lane_counts.get(lane))}" for lane in MIXED_CONVERSATION_REQUIRED_LANES
    )
    detail = (
        f"best recent session: {assistant_turns} assistant turn(s); lanes {lane_bits}; "
        f"chat model {model_count}; fallback {fallback_count}; latency samples {len(latencies)}"
    )
    p95 = _percentile(latencies, 0.95)
    if p95 is not None:
        detail += f"; chat p95 {p95}ms"
    detail += "; metadata-only"

    if assistant_turns < MIXED_CONVERSATION_MIN_ASSISTANT_TURNS:
        needed = MIXED_CONVERSATION_MIN_ASSISTANT_TURNS - assistant_turns
        return (
            None,
            detail
            + f" — not yet proven; collect {needed} more assistant turn(s) in one mixed session"
            + MIXED_CONVERSATION_PROOF_RECOVERY,
        )
    if missing:
        return (
            None,
            detail
            + f" — not yet proven; missing lane(s): {', '.join(missing)}"
            + MIXED_CONVERSATION_PROOF_RECOVERY,
        )
    if model_count == 0:
        return False, detail + " — no model-backed chat in best mixed session"
    if fallback_count > 0 and fallback_count >= model_count:
        return False, detail + " — fallback chat count is too high in best mixed session"
    if len(latencies) < MIXED_CONVERSATION_MIN_CHAT_LATENCY_SAMPLES:
        needed = MIXED_CONVERSATION_MIN_CHAT_LATENCY_SAMPLES - len(latencies)
        return (
            None,
            detail
            + f" — not yet proven; collect {needed} more chat latency sample(s)"
            + MIXED_CONVERSATION_PROOF_RECOVERY,
        )
    if p95 is not None and p95 > MIXED_CONVERSATION_TARGET_P95_MS:
        return (
            False,
            detail
            + f" — chat p95 is above {int(MIXED_CONVERSATION_TARGET_P95_MS)}ms target"
            + MIXED_CONVERSATION_LATENCY_RECOVERY,
        )
    return True, detail + " — mixed-session proof ready for live acceptance review"


def _check_personal_integration_readiness(config) -> tuple[bool | None, str]:
    """Read-only: are calendar/email/reminder tools wired with safe risk gates?"""
    try:
        from jarvis_v2.memory.obsidian import ObsidianVault
        from jarvis_v2.memory.store import MemoryStore
        from jarvis_v2.tools.calendar_connector import _creds_file, _readonly_token_file
        from jarvis_v2.tools.email_connector import _credential_status
        from jarvis_v2.tools.registry import build_core_registry

        registry = build_core_registry(
            MemoryStore(config.db_path),
            ObsidianVault(config.obsidian_vault, config.obsidian_root),
            config,
        )
    except Exception as e:
        return False, f"personal integration registry unreadable: {_safe_exception(e)}"

    missing: list[str] = []
    wrong_risk: list[str] = []
    for tool_name, expected_risk in PERSONAL_INTEGRATION_RISKS.items():
        try:
            tool = registry.get(tool_name)
            actual_risk = _plain_text(getattr(tool.risk, "name", tool.risk))
        except Exception:
            missing.append(tool_name)
            continue
        if actual_risk != expected_risk:
            wrong_risk.append(f"{tool_name}:{actual_risk or 'unknown'}")

    if missing or wrong_risk:
        parts = []
        if missing:
            parts.append(f"missing {len(missing)} tool(s): {', '.join(sorted(missing)[:4])}")
        if wrong_risk:
            parts.append(f"wrong risk {len(wrong_risk)} tool(s): {', '.join(sorted(wrong_risk)[:4])}")
        return False, "; ".join(parts) + " — fix ToolRegistry before live personal-integration proofs"

    try:
        calendar_creds_present = bool(_creds_file().exists())
        calendar_token_present = bool(_readonly_token_file().exists())
    except Exception:
        calendar_creds_present = False
        calendar_token_present = False

    try:
        gmail_status = _credential_status()
        gmail_address_configured = bool(gmail_status.get("address_configured"))
        gmail_password_configured = bool(gmail_status.get("password_configured"))
        gmail_valid = bool(gmail_status.get("valid"))
    except Exception:
        gmail_address_configured = False
        gmail_password_configured = False
        gmail_valid = False

    osascript_present = bool(shutil.which("osascript"))
    calendar_ready = calendar_creds_present and calendar_token_present
    gmail_configured = gmail_address_configured and gmail_password_configured

    calendar_state = "ready" if calendar_ready else "missing OAuth file(s)"
    if gmail_valid:
        gmail_state = "valid"
    elif gmail_configured:
        gmail_state = "configured but invalid"
    else:
        gmail_state = "missing env"
    reminders_state = "osascript present" if osascript_present else "osascript missing"
    detail = (
        f"tool gates ready: {len(PERSONAL_INTEGRATION_RISKS)}/{len(PERSONAL_INTEGRATION_RISKS)}; "
        f"calendar OAuth {calendar_state}; Gmail {gmail_state}; reminders {reminders_state}; "
        "write paths remain EXTERNAL_SIDE_EFFECT or HIGH_RISK and approval-gated; no handlers run"
    )

    if calendar_ready and gmail_valid and osascript_present:
        return True, detail
    return None, detail + " — live proof still needs operator present"


def _check_personal_proof_history(config) -> tuple[bool | None, str]:
    """Read-only: has the audit log captured live personal-integration proofs?

    This reads only bounded tool-run metadata. It does not read calendar/email/
    reminder payloads, and it does not call any personal connector. Contacts
    have their own direct read-only readiness row.
    """
    try:
        from jarvis_v2.memory.store import MemoryStore

        rows = MemoryStore(config.db_path).recent_tool_runs(limit=PERSONAL_PROOF_RECENT_LIMIT)
    except Exception as e:
        return False, f"personal proof audit unreadable: {_safe_exception(e)}"

    reviewed = 0
    failed_or_blocked = 0
    calendar_read = False
    calendar_writes = {tool: False for tool in PERSONAL_PROOF_CALENDAR_WRITE_TOOLS}
    email_read = False
    email_search = False
    email_send = False
    reminder_scheduled = False

    for row in rows:
        tool_name = _plain_text(_mapping_value(row, "tool_name", "")).strip()
        if tool_name not in PERSONAL_PROOF_RELEVANT_TOOLS:
            continue
        reviewed += 1
        ok = _row_bool(row, "ok")
        approved = _row_bool(row, "approved")
        if not ok:
            failed_or_blocked += 1
            continue

        if tool_name in PERSONAL_PROOF_CALENDAR_READ_TOOLS:
            calendar_read = True
        if tool_name in calendar_writes and approved:
            calendar_writes[tool_name] = True
        if tool_name in PERSONAL_PROOF_EMAIL_READ_TOOLS:
            email_read = True
        if tool_name in PERSONAL_PROOF_EMAIL_SEARCH_TOOLS:
            email_search = True
        if tool_name in PERSONAL_PROOF_EMAIL_SEND_TOOLS and approved:
            email_send = True
        if tool_name in PERSONAL_PROOF_REMINDER_TOOLS and approved:
            reminder_scheduled = _reminder_scheduled_proof(_message_metadata(row)) or reminder_scheduled
    write_count = sum(1 for ready in calendar_writes.values() if ready)
    detail = (
        f"reviewed {reviewed} personal audit row(s); calendar read={'yes' if calendar_read else 'no'}, "
        f"calendar write {write_count}/3, email read={'yes' if email_read else 'no'}, "
        f"email search/body={'yes' if email_search else 'no'}, email send={'yes' if email_send else 'no'}, "
        f"approved set_reminder scheduling={'yes' if reminder_scheduled else 'no'}; "
        f"failed/blocked rows {failed_or_blocked}; content suppressed"
    )

    if reviewed == 0:
        return None, detail + " — no personal live-proof audit rows yet; run proofs with operator present"

    missing: list[str] = []
    if not calendar_read:
        missing.append("calendar read")
    if write_count < len(PERSONAL_PROOF_CALENDAR_WRITE_TOOLS):
        missing.append("calendar create/update/delete")
    if not (email_read and email_search):
        missing.append("email read/search")
    if not email_send:
        missing.append("email send")
    if not reminder_scheduled:
        missing.append("approved set_reminder scheduling proof")
    if missing:
        return (
            None,
            detail + f" — not yet proven; missing {', '.join(missing[:6])}; live proof still needs operator present",
        )

    return (
        True,
        detail + " — personal proof history present; reminder phone delivery and target/content details still need operator acceptance review",
    )


def _check_channel_health(config) -> tuple[bool | None, str]:
    """Read-only: can Jarvis summarize messaging/calling channel health safely?"""
    try:
        from jarvis_v2.memory.store import MemoryStore
        from jarvis_v2.tools.channel_health import CHANNELS, MAX_ROWS, WINDOW_DAYS, make_channel_health_tools

        store = MemoryStore(config.db_path)
        tool = make_channel_health_tools(store)[0]
        result = tool({})
    except Exception:
        return False, "channel health unavailable — check local Jarvis storage and retry"
    if not result.ok:
        return False, "channel_health returned not-ok"
    metadata = getattr(result, "metadata", {})
    if (
        _metadata_value(metadata, "content_suppressed") is not True
        or _metadata_value(metadata, "reads_message_content") is not False
    ):
        return False, "channel health metadata did not prove content suppression"

    expected_channels = {channel_id for channel_id, _label, _tools in CHANNELS}
    raw_channels = _metadata_value(metadata, "channels")
    row_count = _metadata_value(metadata, "rows_reviewed")
    channel_count = _metadata_value(metadata, "channel_count")
    if (
        not isinstance(raw_channels, list)
        or type(row_count) is not int
        or row_count < 0
        or row_count > MAX_ROWS
        or type(channel_count) is not int
        or channel_count != len(expected_channels)
        or len(raw_channels) != len(expected_channels)
    ):
        return False, "channel health metadata is malformed"

    recent_successes = 0
    total_success_count = 0
    regressed_channels = 0
    failed_without_success = 0
    seen_channels: set[str] = set()
    for summary in raw_channels:
        if not isinstance(summary, dict):
            return False, "channel health metadata is malformed"
        channel = _plain_text(summary.get("channel")).strip()
        success_count = summary.get("success_count_7d")
        last_success = _plain_text(summary.get("last_success_at")).strip()
        last_failure = _plain_text(summary.get("last_failure_at")).strip()
        if (
            channel not in expected_channels
            or channel in seen_channels
            or type(success_count) is not int
            or success_count < 0
            or success_count > MAX_ROWS
            or not last_success
            or not last_failure
        ):
            return False, "channel health metadata is malformed"
        seen_channels.add(channel)

        success_at = None if last_success == "never" else _parse_safe_datetime(last_success)
        failure_at = None if last_failure == "never" else _parse_safe_datetime(last_failure)
        if (last_success != "never" and success_at is None) or (
            last_failure != "never" and failure_at is None
        ):
            return False, "channel health metadata is malformed"
        if success_count > 0:
            if (
                success_at is None
                or success_at < datetime.now() - timedelta(days=WINDOW_DAYS)
                or success_at > datetime.now() + timedelta(minutes=1)
            ):
                return False, "channel health metadata is malformed"
            recent_successes += 1
            total_success_count += success_count
        if success_at is not None and failure_at is not None and failure_at > success_at:
            regressed_channels += 1
        if success_at is None and failure_at is not None:
            failed_without_success += 1

    if seen_channels != expected_channels:
        return False, "channel health metadata is malformed"
    if total_success_count > row_count:
        return False, "channel health metadata is malformed"

    rows = _safe_count(row_count)
    channels = _safe_count(channel_count)
    bounded = " bounded" if _metadata_value(metadata, "row_sample_truncated") is True else ""
    detail = (
        f"{recent_successes}/{channels} channels with 7d success; {rows} audit row(s) reviewed"
        f"{bounded}, content suppressed"
    )
    if failed_without_success:
        return False, detail + f" — {failed_without_success} channel(s) have failure evidence but no success"
    if recent_successes == 0:
        return None, detail + " — no recent channel success is recorded; run the live channel proof with operator present"
    if regressed_channels:
        return False, detail + f" — {regressed_channels} channel(s) have a newer failure than their last success"
    if recent_successes < len(expected_channels):
        return None, detail + " — remaining channels are unproven or intentionally descoped; record that decision before acceptance"
    return True, "ready; " + detail


def _check_guardrail_control_plane() -> tuple[bool | None, str]:
    """Read-only: can the operator ask why live-channel build paths are frozen?"""
    try:
        from jarvis_v2.agent.command_suggest import suggest_command
        from jarvis_v2.tools.cockpit import CAPABILITY_LANES

        for phrase in (
            "guardrails",
            "show guardrails",
            "freeze status",
            "what is frozen",
            "what can Codex touch",
            "what should you not edit",
            "proofs pending",
            "what is the live test matrix",
            "why frozen",
            "safe lane",
            "what should I test",
            "test matrix status",
            "what live proofs are pending",
            "what live tests are pending",
            "what channels should I test",
            "which channels should I test",
            "what live channels are pending",
            "which channels need live proof",
            "what channels need proof",
            "what results do you need from me",
            "what should operator test",
            "what tests should I run",
            "how should I report live test results",
            "how do I report live test results",
            "live test result format",
            "what format should I use for live test results",
            "what should I send after testing",
            "report live matrix results",
            "report live test results",
            "프리즈 상태",
            "동결 파일",
            "검증 대기 뭐야",
            "테스트 매트릭스 보여줘",
            "라이브 테스트 뭐 해야 해",
            "어떤 채널 테스트해",
            "라이브 채널 뭐 테스트해",
            "테스트 매트릭스 상태",
            "라이브 증명 상태",
            "수정 금지 파일",
            "동결 상태",
        ):
            suggestion = suggest_command(phrase)
            if not suggestion or "cockpit" not in suggestion.casefold():
                return False, "guardrail/freeze phrases do not redirect to cockpit"
        lane = next(
            (lane for lane in CAPABILITY_LANES if _plain_text(lane.get("key")).casefold() == "build_guardrails"),
            None,
        )
    except Exception as e:
        return False, f"EXCEPTION: {_safe_exception(e)}"
    if not isinstance(lane, dict):
        return False, "Build Guardrails cockpit lane is missing"
    tools = {_plain_text(tool).casefold() for tool in (lane.get("tools") or [])}
    if "capability_cockpit" not in tools:
        return False, "Build Guardrails lane no longer points at capability_cockpit"
    notes = [_plain_text(note).casefold() for note in (lane.get("notes") or [])]
    if not any("live-proof freeze active" in note for note in notes):
        return False, "Build Guardrails lane no longer explains the live-proof freeze"
    if not any("planner/send/call/hud" in note for note in notes):
        return False, "Build Guardrails lane no longer names the frozen planner/send/call/HUD paths"
    if not any(
        "pending live-proof matrix" in note
        and "kakao" in note
        and "instagram" in note
        and "telegram" in note
        and "imessage" in note
        and "facetime" in note
        for note in notes
    ):
        return False, "Build Guardrails lane no longer names the pending live-proof channel categories"
    if not any(
        "report live matrix results" in note
        and "pass/fail" in note
        and "stage/error" in note
        and "message content" in note
        for note in notes
    ):
        return False, "Build Guardrails lane no longer explains the safe live-result reporting format"
    if not any("local-safe" in note for note in notes):
        return False, "Build Guardrails lane no longer explains the autonomous safe lane"
    return True, "ready; guardrails/freeze/live-test-result/live-result-reporting/result-format asks redirect to cockpit and explain the live-proof freeze plus safe reporting format for planner/send/call/HUD paths"


def _check_proof_ledger_control_plane() -> tuple[bool | None, str]:
    """Read-only: do proof/evidence asks reach existing evidence surfaces?"""
    try:
        from jarvis_v2.agent.command_suggest import suggest_command

        cases = (
            ("show proof", "evidence ledger"),
            ("what proof do you have", "evidence ledger"),
            ("evidence status", "evidence ledger"),
            ("trust evidence", "evidence ledger"),
            ("completion evidence", "evidence ledger"),
            ("latest verification proof", "verification receipt latest"),
            ("증거상태", "evidence ledger"),
            ("증거보여줘", "evidence ledger"),
            ("검증증거", "verification receipt latest"),
            ("최근검증", "verification receipt latest"),
        )
        for phrase, expected in cases:
            suggestion = suggest_command(phrase)
            if not suggestion or expected not in suggestion.casefold():
                return False, "proof/evidence phrases do not redirect to trusted proof surfaces"
    except Exception as e:
        return False, f"EXCEPTION: {_safe_exception(e)}"
    return True, "ready; proof and evidence asks redirect to read-only ledger and receipt surfaces"


def _check_completion_phone_gate() -> tuple[bool | None, str]:
    """Read-only: do phone completion questions reach the evidence gate?"""
    try:
        from jarvis_v2.automations.telegram_control import TelegramCommandBridge

        class _Runtime:
            def __init__(self) -> None:
                self.inputs: list[str] = []

            def handle(self, text: str, request_token: str | None = None):
                self.inputs.append(text)
                return SimpleNamespace(
                    response=(
                        "Completion claim gate: BLOCKED until live channel proofs "
                        "and recovery/learning proof debt are reviewed."
                    ),
                    tool_results=[],
                )

        runtime = _Runtime()
        bridge = TelegramCommandBridge(
            runtime_factory=lambda: runtime,
            send_func=lambda _chat_id, _text, _markup=None: {"ok": True},
            fetch_func=lambda _token, _offset, _timeout: [],
            chat_action_func=None,
        )
        phrases = (
            "is Jarvis done?",
            "are you done?",
            "completion status",
            "completion claim gate",
            "can Jarvis claim completion?",
            "can Jarvis claim done?",
            "jarvis completion status",
            "why isn't Jarvis done?",
            "what blocks completion?",
            "what is blocking completion?",
            "when is Jarvis done?",
            "자비스 끝났어?",
            "자비스 완료상태",
            "완료 뭐 막혀",
            "자비스 왜 아직 안 끝났어?",
            "완료 주장 게이트",
        )
        for phrase in phrases:
            reply, markup = bridge._handle_command(phrase)
            if markup is not None:
                return False, "completion phone gate unexpectedly returned approval buttons"
            if not runtime.inputs or runtime.inputs[-1] != "completion claim gate":
                return False, f"completion phone phrase {phrase!r} did not route to the claim gate"
            if "Completion claim gate" not in reply or "BLOCKED" not in reply:
                return False, "completion phone gate lost the blocked evidence signal"
    except Exception as e:
        return False, f"EXCEPTION: {_safe_exception(e)}"
    return True, "ready; completion phone aliases route to the claim gate with blocked evidence and no approval buttons"


def _check_learning_recovery_control_plane() -> tuple[bool | None, str]:
    """Read-only: do learning/recovery asks reach existing proof surfaces?"""
    try:
        from jarvis_v2.agent.command_suggest import suggest_command

        cases = (
            ("learning status", "learning review"),
            ("learning loop status", "learning review"),
            ("what did you learn", "learning review"),
            ("학습상태", "learning review"),
            ("뭘배웠어", "learning review"),
            ("recovery status", "recovery closure checklist"),
            ("recovery plan", "recovery closure checklist"),
            ("복구상태", "recovery closure checklist"),
            ("복구계획", "recovery closure checklist"),
        )
        for phrase, expected in cases:
            suggestion = suggest_command(phrase)
            if not suggestion or expected not in suggestion.casefold():
                return False, "learning/recovery phrases do not redirect to trusted proof surfaces"
    except Exception as e:
        return False, f"EXCEPTION: {_safe_exception(e)}"
    return True, "ready; learning and recovery asks redirect to read-only proof surfaces"


def _check_phone_approval_flow() -> tuple[bool | None, str]:
    """Read-only: do owner-phone approval buttons/callbacks keep the safe route?"""
    try:
        from jarvis_v2.automations.telegram_control import TelegramCommandBridge

        owner = "555001"

        class _Runtime:
            def __init__(self) -> None:
                self.inputs: list[str] = []

            def handle(self, text: str, request_token: str | None = None):
                self.inputs.append(text)
                replies = {
                    "queue risky action": "Safety receipt: queued as approval #7",
                    "approval packet 7": "Approval packet ready.",
                    "approve approval 7": "Approved approval #7.",
                    "dismiss approval 7": "Dismissed approval #7.",
                }
                return SimpleNamespace(response=replies.get(text, "Unexpected route."), tool_results=[])

        def _callback(data: str, *, user: str = owner, chat: str = owner, message_id: int = 90) -> dict[str, object]:
            return {
                "id": f"callback-{message_id}",
                "from": {"id": user},
                "data": data,
                "message": {"message_id": message_id, "chat": {"id": chat}},
            }

        prompt_runtime = _Runtime()
        prompt_bridge = TelegramCommandBridge(
            runtime_factory=lambda: prompt_runtime,
            send_func=lambda _chat_id, _text, _markup=None: {"ok": True},
            fetch_func=lambda _token, _offset, _timeout: [],
            answer_callback_func=lambda _callback_id, _text: {"ok": True},
            edit_message_func=lambda _chat_id, _message_id, _text, _markup=None: {"ok": True},
            chat_action_func=None,
        )
        reply, markup = prompt_bridge._handle_command("queue risky action")
        buttons = (markup or {}).get("inline_keyboard", [[]])[0]
        callbacks = [button.get("callback_data") for button in buttons]
        if callbacks != ["approve:7", "deny:7"]:
            return False, "phone approval buttons missing or malformed"
        if "queued as approval" not in reply:
            return False, "phone approval prompt lost the approval receipt"
        if prompt_runtime.inputs != ["queue risky action"]:
            return False, "phone approval prompt route drift"

        approve_runtime = _Runtime()
        approve_answered: list[tuple[str, str]] = []
        approve_edited: list[tuple[str, int, str, object]] = []
        approve_bridge = TelegramCommandBridge(
            runtime_factory=lambda: approve_runtime,
            send_func=lambda _chat_id, _text, _markup=None: {"ok": True},
            fetch_func=lambda _token, _offset, _timeout: [],
            answer_callback_func=lambda callback_id, text: approve_answered.append((callback_id, text)) or {"ok": True},
            edit_message_func=lambda chat_id, message_id, text, markup=None: approve_edited.append(
                (chat_id, message_id, text, markup)
            )
            or {"ok": True},
            chat_action_func=None,
        )
        if not approve_bridge._process_callback(_callback("approve:7"), owner):
            return False, "owner approve callback was not processed"
        if approve_runtime.inputs != ["approval packet 7", "approve approval 7"]:
            return False, "owner approve callback route drift"
        if not approve_answered or not approve_edited:
            return False, "owner approve callback did not update the phone message"

        deny_runtime = _Runtime()
        deny_edited: list[tuple[str, int, str, object]] = []
        deny_bridge = TelegramCommandBridge(
            runtime_factory=lambda: deny_runtime,
            send_func=lambda _chat_id, _text, _markup=None: {"ok": True},
            fetch_func=lambda _token, _offset, _timeout: [],
            answer_callback_func=lambda _callback_id, _text: {"ok": True},
            edit_message_func=lambda chat_id, message_id, text, markup=None: deny_edited.append(
                (chat_id, message_id, text, markup)
            )
            or {"ok": True},
            chat_action_func=None,
        )
        if not deny_bridge._process_callback(_callback("deny:7", message_id=91), owner):
            return False, "owner deny callback was not processed"
        if deny_runtime.inputs != ["dismiss approval 7"]:
            return False, "owner deny callback route drift"
        if not deny_edited:
            return False, "owner deny callback did not update the phone message"

        blocked_runtime = _Runtime()
        blocked_bridge = TelegramCommandBridge(
            runtime_factory=lambda: blocked_runtime,
            send_func=lambda _chat_id, _text, _markup=None: {"ok": True},
            fetch_func=lambda _token, _offset, _timeout: [],
            answer_callback_func=lambda _callback_id, _text: {"ok": True},
            edit_message_func=lambda _chat_id, _message_id, _text, _markup=None: {"ok": True},
            chat_action_func=None,
        )
        if blocked_bridge._process_callback(_callback("approve:7", user="999999", message_id=92), owner):
            return False, "non-owner approval callback was processed"
        if blocked_runtime.inputs:
            return False, "non-owner approval callback reached the runtime"
    except Exception as e:
        return False, f"EXCEPTION: {_safe_exception(e)}"
    return (
        True,
        (
            "ready; phone approval buttons and owner approve/deny callbacks route through the fake runtime; "
            "non-owner callback ignored; read-only local check only, no live approval"
        ),
    )


def _check_phone_control_center() -> tuple[bool | None, str]:
    """Read-only: do owner-phone status shortcuts route to trusted surfaces?"""
    try:
        from jarvis_v2.automations.telegram_control import TelegramCommandBridge

        class _CockpitTool:
            def handler(self, _args):
                return SimpleNamespace(
                    output=(
                        "Capability cockpit: lanes ready. "
                        "Acceptance Harness: live_check publishes the one-screen acceptance table; "
                        "readiness rows do not replace live delivery, approval, phone, reboot, or daily-streak proof. "
                        "Build Guardrails: live-proof freeze active for planner/send/call/HUD files. "
                        "pending live-proof matrix covers Kakao, Instagram, Telegram, iMessage, phone, and FaceTime channel checks. "
                        "This is read-only."
                    ),
                    metadata={
                        "lane_count": 5,
                        "attention_lane_count": 1,
                        "status_counts": {"attention": 1, "awaiting approval": 1, "ok": 3},
                        "trust_summary": {
                            "summary": "trust checklist 4/4 lanes, 12/12 checks ready",
                            "non_authorizing": True,
                        },
                        "proof_lane_count": 3,
                        "proof_point_count": 9,
                        "proof_summary": [
                            {
                                "lane_key": "operator_workflow_evals",
                                "lane_title": "Operator Workflow Evals",
                                "proof_count": 4,
                                "sample_proofs": [
                                    "Korean Telegram send stops at approval with Hangul recipient/body intact",
                                    "Morning Brief can be pushed on demand to the owner phone channel",
                                ],
                            }
                        ],
                        "proof_summary_hidden_lane_count": 2,
                        "proof_summary_hidden_lanes": [
                            {"lane_key": "learning_loop", "lane_title": "Learning Loop", "proof_count": 4},
                            {
                                "lane_key": "internal_orchestration",
                                "lane_title": "Internal Orchestration",
                                "proof_count": 3,
                            },
                        ],
                        "next_command_queue": [
                            {
                                "command": "channel health",
                                "kind": "diagnostic",
                                "lane_title": "Messaging",
                                "status": "attention",
                            },
                            {
                                "command": "approval readiness 7",
                                "kind": "approval_review",
                                "lane_title": "Approvals",
                                "status": "awaiting approval",
                            }
                        ],
                        "capability_lanes": [
                            {
                                "key": "messaging",
                                "title": "Messaging",
                                "status": "attention",
                                "next_command": "channel health",
                                "attention_reasons": ["transport_error"],
                                "last_failure": "transport failed",
                                "last_failure_kind": "transport_error",
                            },
                            {
                                "key": "approvals",
                                "title": "Approvals",
                                "status": "awaiting approval",
                                "next_command": "approval readiness 7",
                                "last_approval_hold": "held at approval gate",
                                "approval_next_commands": ["approval readiness 7", "approval packet 7"],
                            },
                            {
                                "key": "voice",
                                "title": "Voice",
                                "status": "ok",
                                "next_command": "voice setup check",
                            },
                            {
                                "key": "orchestration",
                                "title": "Internal Orchestration",
                                "status": "ready",
                                "risk": "READ_ONLY",
                                "approval_required": False,
                                "tool_coverage": "registered",
                                "smoke_coverage": "registered",
                                "next_command": "jarvis status",
                                "next_command_kind": "example",
                                "example_command": "jarvis status",
                                "guardrail_notes": ["internal subagents are workers, not companion personas"],
                            },
                            {
                                "key": "acceptance_harness",
                                "title": "Acceptance Harness",
                                "status": "ready",
                                "risk": "READ_ONLY",
                                "approval_required": False,
                                "tool_coverage": "registered",
                                "smoke_coverage": "registered",
                                "next_command": "live proof status",
                                "next_command_kind": "example",
                                "example_command": "live proof status",
                                "guardrail_notes": [
                                    "diagnostic only; run opt-in live_check with operator present for real acceptance proof",
                                    "readiness rows do not replace live delivery, approval, phone, reboot, or daily-streak proof",
                                ],
                                "proof_points": [
                                    "live_check publishes the one-screen acceptance table",
                                    "acceptance gaps row counts open Definition-of-Done boxes without echoing private targets",
                                    "acceptance gaps points to acceptance next for prioritized proof guidance without exposing checklist text",
                                    "acceptance next row names the next safe proof lane without running live checks",
                                    "aggregate smoke row reads the latest logged smoke_test_all proof without running tests",
                                    "personal proofs row audits bounded metadata only",
                                    "daemon startup row checks launcher contracts, LaunchAgent templates, scheduler ticker ownership, and restart documentation without process control",
                                ],
                            },
                            {
                                "key": "personal_proofs",
                                "title": "Personal Proofs",
                                "status": "ready",
                                "risk": "HIGH_RISK",
                                "approval_required": True,
                                "tool_coverage": "registered",
                                "smoke_coverage": "registered",
                                "next_command": "personal proofs status",
                                "next_command_kind": "example",
                                "example_command": "personal proofs status",
                                "guardrail_notes": [
                                    "read-only proof lane for WS2: calendar writes, email read/search/send, set_reminder delivery, and contact lookup",
                                    "run opt-in live_check with operator present for actual proof; this lane does not access accounts or send anything",
                                ],
                                "proof_points": [
                                    "personal proofs are still live-acceptance evidence, not mocked completion claims",
                                    "set_reminder proof uses Jarvis's local Telegram reminder path",
                                ],
                            },
                            {
                                "key": "research_web",
                                "title": "Research & Web",
                                "status": "ready",
                                "risk": "LOCAL_SAFE",
                                "approval_required": False,
                                "tool_coverage": "registered",
                                "smoke_coverage": "registered",
                                "next_command": "research Zoey OS",
                                "next_command_kind": "example",
                                "example_command": "research Zoey OS",
                                "guardrail_notes": [
                                    "research may call external search/page services and the local model only when explicitly run",
                                    "the cockpit lane itself is read-only and does not fetch pages, synthesize answers, write notes, or queue approvals",
                                ],
                                "proof_points": [
                                    "research and web_lookup emit no-authority handoff packets with content excluded from metadata",
                                    "live_check verifies research/web registration, risk gates, and route shape without fetching the web or calling models",
                                ],
                            },
                            {
                                "key": "build_guardrails",
                                "title": "Build Guardrails",
                                "status": "ready",
                                "risk": "READ_ONLY",
                                "approval_required": False,
                                "tool_coverage": "registered",
                                "smoke_coverage": "registered",
                                "next_command": "cockpit",
                                "next_command_kind": "diagnostic",
                                "example_command": "cockpit",
                                "guardrail_notes": [
                                    "live-proof freeze active for planner/send/call/HUD files until the operator posts channel test results",
                                    "pending live-proof matrix covers Kakao, Instagram, Telegram, iMessage, phone, and FaceTime channel checks",
                                    "report live matrix results as channel, pass/fail, and last visible stage/error; do not include secrets or message content",
                                    "autonomous passes stay local-safe: diagnostics, handoff accuracy, docs, readiness reports, adjacent smokes",
                                ],
                            },
                            {
                                "key": "operator_workflow_evals",
                                "title": "Operator Workflow Evals",
                                "status": "ready",
                                "risk": "HIGH_RISK",
                                "approval_required": True,
                                "tool_coverage": "registered",
                                "smoke_coverage": "registered",
                                "next_command": "channel health",
                                "next_command_kind": "diagnostic",
                                "example_command": "push today's brief to my phone",
                                "attention_reasons": [
                                    "pins Korean sends, phone brief, channel diagnostics, contacts, markets"
                                ],
                                "proof_points": [
                                    "Korean Telegram send stops at approval with Hangul recipient/body intact",
                                    "Morning Brief can be pushed on demand to the owner phone channel",
                                    "Phone control shortcuts expose status, failures, channels, approvals, cockpit, lanes, and guardrails",
                                    "Clean-state cockpit has zero false attention lanes",
                                ],
                            },
                        ],
                    },
                )

        class _Registry:
            def get(self, name: str):
                if name != "capability_cockpit":
                    raise KeyError(name)
                return _CockpitTool()

        class _Runtime:
            def __init__(self) -> None:
                self.inputs: list[str] = []
                self.registry = _Registry()

            def handle(self, text: str, request_token: str | None = None):
                self.inputs.append(text)
                replies = {
                    "jarvis status": "Jarvis status: ready in the local-safe phone control check.",
                    "voice command cockpit": "Jarvis voice command cockpit: read-only route gate is available.",
                    "voice setup check": "Jarvis voice setup check: read-only, no microphone access requested.",
                    "voice stop intent: stop listening": "Jarvis voice stop intent packet: read-only stop or rerecord intent.",
                    "capability map": "Jarvis capability map: safe commands and approval-gated actions are visible.",
                    "build progress": "Jarvis build progress report: harness readiness remains visible.",
                    "roadmap": "Jarvis roadmap: next assistant layers are visible.",
                    "agi gates": "AGI gates: harness readiness gates are reviewable.",
                    "list goals": "Active goals: Jarvis durable goals are visible.",
                    "subagent fleet status": (
                        "Internal worker fleet status: ready workers are visible as internal capacity."
                    ),
                    "work queue": (
                        "Jarvis work queue: Zoey/OpenClaw/Hermes research points Jarvis "
                        "toward a visible control plane, not companion personas. "
                        "Other Jarvis-style systems reinforce visibility before broad autonomy."
                    ),
                    "safety status": "Jarvis safety status: approvals guard risky actions.",
                    "privacy report": "Jarvis privacy report: private-data boundaries are visible.",
                    "risk matrix": "Jarvis risk matrix: READ_ONLY and HIGH_RISK tools are separated.",
                    "recent tool runs": "Recent tool runs: no failures in the bounded sample.",
                    "list scheduled jobs": "Scheduled jobs: Morning Brief is configured.",
                    "channel health": "channel health: read-only channel diagnostics are available.",
                    "pending approvals": "Pending approvals: none.",
                    "readiness report": "Readiness report: local-safe surfaces are ready.",
                    "memory stats": "Memory stats: local memory/state counters are visible.",
                    "learning review": "Learning review: learning-loop proof debt is visible.",
                    "model routing status": (
                        "Jarvis model routing status: Conversation latency proof is visible "
                        "with JARVIS_CHAT_MAX_REPLY_TOKENS and JARVIS_CHAT_MAX_HISTORY_MESSAGES."
                    ),
                    "completion_next_proof_packet": (
                        "Jarvis completion next proof packet: read-only next safe proof lane is visible."
                    ),
                    "setup check": "Jarvis setup check: optional dependency readiness is visible.",
                    "jarvis doctor": "Jarvis doctor: deeper local diagnostics are available.",
                }
                return SimpleNamespace(response=replies.get(text, f"Unexpected route: {text}"), tool_results=[])

        runtime = _Runtime()
        bridge = TelegramCommandBridge(
            runtime_factory=lambda: runtime,
            send_func=lambda _chat_id, _text, _markup=None: {"ok": True},
            fetch_func=lambda _token, _offset, _timeout: [],
            chat_action_func=None,
        )
        routed_cases = (
            ("status", "jarvis status", "Jarvis status"),
            ("상태 확인", "jarvis status", "Jarvis status"),
            ("/voice", "voice command cockpit", "voice command cockpit"),
            ("음성", "voice command cockpit", "voice command cockpit"),
            ("voice setup", "voice setup check", "voice setup check"),
            ("마이크 확인", "voice setup check", "voice setup check"),
            ("voice stop", "voice stop intent: stop listening", "voice stop intent packet"),
            ("음성 중지", "voice stop intent: stop listening", "voice stop intent packet"),
            ("capabilities", "capability map", "capability map"),
            ("what can you do?", "capability map", "capability map"),
            ("기능 알려줘", "capability map", "capability map"),
            ("명령어 알려줘", "capability map", "capability map"),
            ("자비스 뭐 할 수 있어", "capability map", "capability map"),
            ("AGI progress", "build progress", "build progress report"),
            ("build status", "build progress", "build progress report"),
            ("자비스 진행상황", "build progress", "build progress report"),
            ("roadmap", "roadmap", "roadmap"),
            ("what should Jarvis build next?", "roadmap", "roadmap"),
            ("로드맵 보여줘", "roadmap", "roadmap"),
            ("AGI status", "agi gates", "AGI gates"),
            ("harness readiness", "agi gates", "AGI gates"),
            ("AGI 준비 상태", "agi gates", "AGI gates"),
            ("goals", "list goals", "Active goals"),
            ("goal status", "list goals", "Active goals"),
            ("목표 상태", "list goals", "Active goals"),
            ("agent research note", "work queue", "Zoey/OpenClaw/Hermes research"),
            ("AI agent landscape", "work queue", "visible control plane"),
            ("three month plan", "work queue", "visible control plane"),
            ("three months of work", "work queue", "visible control plane"),
            ("Jarvis three month plan", "work queue", "visible control plane"),
            ("what are the three months of work", "work queue", "visible control plane"),
            ("what is the three month plan for Jarvis", "work queue", "visible control plane"),
            ("what should Jarvis focus on for the next three months", "work queue", "visible control plane"),
            ("how do we unlock the moat", "work queue", "visible control plane"),
            ("what unlocks Jarvis moat", "work queue", "visible control plane"),
            ("what should Jarvis build before integrations", "work queue", "visible control plane"),
            ("when should Jarvis add integrations", "work queue", "visible control plane"),
            ("should Jarvis add integrations now", "work queue", "visible control plane"),
            ("other Jarvis models", "work queue", "Other Jarvis-style systems"),
            ("what did you find about Zoey?", "work queue", "visible control plane"),
            ("what did you find about other Jarvis models", "work queue", "Other Jarvis-style systems"),
            ("what should Jarvis copy from OpenClaw", "work queue", "visible control plane"),
            ("what should Jarvis avoid from Zoey", "work queue", "not companion personas"),
            ("what is the Jarvis strategy", "work queue", "visible control plane"),
            ("Jarvis product strategy", "work queue", "visible control plane"),
            ("what is the right move for Jarvis", "work queue", "visible control plane"),
            ("what should we do after Zoey research?", "work queue", "visible control plane"),
            ("should Jarvis use companions", "work queue", "not companion personas"),
            ("should Jarvis copy Zoey?", "work queue", "not companion personas"),
            ("what is Jarvis moat", "work queue", "visible control plane"),
            ("why no companion personas", "work queue", "not companion personas"),
            ("Lindy research", "work queue", "visible control plane"),
            ("Manus research", "work queue", "visible control plane"),
            ("조이 조사 결과", "work queue", "not companion personas"),
            ("다른 자비스 모델", "work queue", "Other Jarvis-style systems"),
            ("자비스 차별점", "work queue", "visible control plane"),
            ("자비스 3개월 계획", "work queue", "visible control plane"),
            ("자비스 세 달 계획", "work queue", "visible control plane"),
            ("자비스 다음 3개월 뭐 해", "work queue", "visible control plane"),
            ("자비스 해자 어떻게 열어", "work queue", "visible control plane"),
            ("통합 지금 추가해도 돼", "work queue", "visible control plane"),
            ("통합 언제 추가해", "work queue", "visible control plane"),
            ("자비스 전략", "work queue", "visible control plane"),
            ("자비스 방향", "work queue", "visible control plane"),
            ("조이 이후 뭐 만들까", "work queue", "visible control plane"),
            ("자비스 조이 이후 뭐 만들어", "work queue", "visible control plane"),
            ("조이 따라해야 해", "work queue", "not companion personas"),
            ("자비스 컴패니언 해야 해", "work queue", "not companion personas"),
            ("왜 컴패니언 안 해", "work queue", "not companion personas"),
            ("/safety", "safety status", "safety status"),
            ("안전", "safety status", "safety status"),
            ("/privacy", "privacy report", "privacy report"),
            ("개인정보", "privacy report", "privacy report"),
            ("/risk", "risk matrix", "risk matrix"),
            ("위험", "risk matrix", "risk matrix"),
            ("what broke", "recent tool runs", "Recent tool runs"),
            ("뭐가 고장났어", "recent tool runs", "Recent tool runs"),
            ("brief status", "list scheduled jobs", "Scheduled jobs"),
            ("브리핑 확인", "list scheduled jobs", "Scheduled jobs"),
            ("channels", "channel health", "channel health"),
            ("did it send?", "channel health", "channel health"),
            ("was it sent?", "channel health", "channel health"),
            ("message sent?", "channel health", "channel health"),
            ("did telegram send?", "channel health", "channel health"),
            ("telegram delivered?", "channel health", "channel health"),
            ("did the message send?", "channel health", "channel health"),
            ("did the call go through?", "channel health", "channel health"),
            ("채널 상태", "channel health", "channel health"),
            ("채널 헬스", "channel health", "channel health"),
            ("메시지 보내졌어?", "channel health", "channel health"),
            ("텔레그램 보내졌어?", "channel health", "channel health"),
            ("카톡 보내졌어?", "channel health", "channel health"),
            ("아이메시지 갔어?", "channel health", "channel health"),
            ("인스타그램 보냈어?", "channel health", "channel health"),
            ("텔레그램 보내졌나요?", "channel health", "channel health"),
            ("카톡 갔나요?", "channel health", "channel health"),
            ("전화 연결됐어?", "channel health", "channel health"),
            ("전화 연결됬나요?", "channel health", "channel health"),
            ("통화 연결됐어?", "channel health", "channel health"),
            ("페이스타임 연결됐어?", "channel health", "channel health"),
            ("approvals", "pending approvals", "Pending approvals"),
            ("승인 대기", "pending approvals", "Pending approvals"),
            ("readiness", "readiness report", "Readiness report"),
            ("준비 상태", "readiness report", "Readiness report"),
            ("memory status", "memory stats", "Memory stats"),
            ("기억 상태", "memory stats", "Memory stats"),
            ("learning status", "learning review", "Learning review"),
            ("학습 상태", "learning review", "Learning review"),
            ("what did you learn?", "learning review", "Learning review"),
            ("뭘 배웠어?", "learning review", "Learning review"),
            ("make chat faster", "model routing status", "Conversation latency proof"),
            ("chat token cap", "model routing status", "Conversation latency proof"),
            ("chat reply token cap", "model routing status", "Conversation latency proof"),
            ("Jarvis speed status", "model routing status", "Conversation latency proof"),
            ("채팅 속도 설정", "model routing status", "Conversation latency proof"),
            ("setup", "setup check", "setup check"),
            ("설정 확인", "setup check", "setup check"),
            ("doctor", "jarvis doctor", "Jarvis doctor"),
            ("진단", "jarvis doctor", "Jarvis doctor"),
        )
        for phrase, routed_to, expected in routed_cases:
            reply, markup = bridge._handle_command(phrase)
            if markup is not None:
                return False, "phone control status shortcut unexpectedly returned approval buttons"
            if not runtime.inputs or runtime.inputs[-1] != routed_to:
                return False, "phone control status shortcuts do not route to trusted diagnostics"
            if expected not in reply:
                return False, "phone control status shortcut lost its diagnostic reply"

        before_help = len(runtime.inputs)
        help_reply, help_markup = bridge._handle_command("/help")
        if help_markup is not None:
            return False, "phone control help unexpectedly returned approval buttons"
        if len(runtime.inputs) != before_help:
            return False, "phone control help should not route through runtime.handle"
        help_lower = help_reply.casefold()
        for expected in (
            "trust checklist",
            "earned trust checklist",
            "what can you do",
            "voice lane",
            "voice capability health",
            "voice proof status",
            "open voice acceptance proof gaps",
            "voice proof matrix",
            "voice live-proof report format",
            "no transcript contents",
            "mixed conversation proof matrix",
            "10+ turn chat/research/tool proof report format",
            "no transcript contents",
            "what should I test?",
            "pending live-proof matrix",
            "live test result format",
            "pass/fail",
            "stage/error",
            "message content",
            "personal proof matrix",
            "personal integration proof report format",
            "no private account details",
            "scheduler proof matrix",
            "scheduled-job 7-day proof report format",
            "no private payloads",
            "approval proof matrix",
            "phone approval-flow proof report format",
            "no private planned args",
            "phone control proof matrix",
            "what broke/cockpit/approvals proof report format",
            "no private run payloads",
            "폰컨트롤 증명 매트릭스",
            "폰 제어 증명 형식",
            "reboot proof matrix",
            "daemon/reboot/network recovery proof report format",
            "no private logs",
            "재부팅 증명 매트릭스",
            "데몬 증명 매트릭스",
            "agi progress",
            "roadmap",
            "agi status",
            "goals",
            "worker status",
            "proofs",
            "what broke",
            "brief status",
            "channels",
            "approvals lane",
            "pending approval review health",
            "calls lane",
            "call capability health",
            "morning brief lane",
            "contacts lane",
            "calendar email lane",
            "personal proofs lane",
            "personal integration proof gaps",
            "markets lane",
            "weather lane",
            "markets/weather capability health",
            "research lane",
            "web lane",
            "research/web lookup capability health",
            "memory lane",
            "memory/notes capability health",
            "diagnostics lane",
            "local diagnostics/recovery health",
            "operator evals lane",
            "workflow evals lane",
            "operator workflow eval proof lane",
            "scheduler lane",
            "live proof status",
            "acceptance harness status",
            "acceptance gaps",
            "open August acceptance gaps",
            "acceptance next for prioritized proof guidance",
            "acceptance next proof",
            "next safe proof lane",
            "smoke status",
            "latest aggregate smoke proof",
            "reboot status",
            "daemon startup and reboot proof readiness",
            "did it send?",
            "memory status",
            "learning status",
            "completion status",
            "why isn't Jarvis done?",
            "completion blockers and proof debt",
            "신뢰 체크리스트",
            "믿어도 되는 이유",
            "기능 알려줘",
            "라이브 테스트 뭐 해야 해",
            "라이브 결과 형식",
            "개인 증명 매트릭스",
            "개인 증명 결과 형식",
            "스케줄 증명 매트릭스",
            "예약 작업 증명 형식",
            "승인 증명 매트릭스",
            "승인 결과 형식",
            "대화 증명 매트릭스",
            "대화 증명 결과 형식",
            "다음 증명",
            "다음 라이브 증명",
            "스모크 상태",
            "테스트 초록",
            "재부팅 상태",
            "데몬 상태",
            "agi 진행상황",
            "로드맵",
            "agi 상태",
            "목표 상태",
            "워커 상태",
            "자비스 끝났어?",
            "완료 뭐 막혀",
            "음성 레인",
            "마이크 레인",
            "음성 증명 상태",
            "한국어 음성 증명",
            "음성 증명 매트릭스",
            "음성 증명 결과 형식",
            "증거 상태",
            "뭐가 고장났어",
            "승인 레인",
            "통화 레인",
            "전화 레인",
            "브리핑 레인",
            "연락처 레인",
            "캘린더 이메일 레인",
            "개인 증명 레인",
            "시장 레인",
            "날씨 레인",
            "검색 레인",
            "연구 레인",
            "기억 레인",
            "진단 레인",
            "제임스 평가 레인",
            "워크플로우 평가 레인",
            "스케줄 레인",
            "라이브 증명 상태",
            "인수 상태",
            "남은 기준",
            "뭐 남았어",
            "기억 상태",
            "학습 상태",
            "채널 상태",
            "전화 연결됐어?",
        ):
            if expected.casefold() not in help_lower:
                return False, f"phone control help lost discovery text for {expected}"
        for stale in (
            "agent status — internal subagent/worker readiness",
            "• agent status",
            "에이전트 상태",
        ):
            if stale.casefold() in help_lower:
                return False, "phone control help promoted stale external agent wording"

        direct_cases = (
            ("cockpit summary", "Cockpit summary"),
            ("콕핏 요약", "Cockpit summary"),
            ("trust checklist", "Cockpit summary"),
            ("earned trust checklist", "Cockpit summary"),
            ("why should I trust Jarvis?", "Cockpit summary"),
            ("what makes Jarvis trustworthy?", "Cockpit summary"),
            ("신뢰 체크리스트", "Cockpit summary"),
            ("왜 자비스 믿어도 돼", "Cockpit summary"),
            ("믿어도 되는 이유", "Cockpit summary"),
            ("next action", "Next cockpit action"),
            ("다음 행동", "Next cockpit action"),
            ("attention", "Cockpit attention"),
            ("주의 상태", "Cockpit attention"),
            ("approvals lane", "Cockpit lane: Approvals"),
            ("승인 상태", "Cockpit lane: Approvals"),
            ("worker status", "Cockpit lane:"),
            ("워커 상태", "Cockpit lane:"),
            ("proofs", "Cockpit proofs:"),
            ("evidence", "Cockpit proofs:"),
            ("what proof do we have?", "Cockpit proofs:"),
            ("증거 상태", "Cockpit proofs:"),
            ("trust tests", "Cockpit lane: Operator Workflow Evals"),
            ("operator real workflow tests", "Cockpit lane: Operator Workflow Evals"),
            ("eval pack plan", "Cockpit lane: Operator Workflow Evals"),
            ("what is the eval pack", "Cockpit lane: Operator Workflow Evals"),
            ("what should Jarvis prove live", "Cockpit lane: Operator Workflow Evals"),
            ("is Korean messaging tested", "Cockpit lane: Operator Workflow Evals"),
            ("Korean message proof", "Cockpit lane: Operator Workflow Evals"),
            ("can Jarvis safely send Korean messages", "Cockpit lane: Operator Workflow Evals"),
            ("한국어 메시지 평가", "Cockpit lane: Operator Workflow Evals"),
            ("한국어 메시지 증명", "Cockpit lane: Operator Workflow Evals"),
            ("한글 전송 증명", "Cockpit lane: Operator Workflow Evals"),
            ("가상연락처이 전송 테스트", "Cockpit lane: Operator Workflow Evals"),
            ("모닝브리핑 평가", "Cockpit lane: Operator Workflow Evals"),
            ("freeze status", "Capability cockpit"),
            ("live proof status", "Cockpit lane: Acceptance Harness"),
            ("acceptance harness status", "Cockpit lane: Acceptance Harness"),
            ("live_check status", "Cockpit lane: Acceptance Harness"),
            ("live_check coverage", "Cockpit lane: Acceptance Harness"),
            ("live_check rows", "Cockpit lane: Acceptance Harness"),
            ("live_check missing rows", "Cockpit lane: Acceptance Harness"),
            ("one screen live_check table", "Cockpit lane: Acceptance Harness"),
            ("does live_check cover every checklist section", "Cockpit lane: Acceptance Harness"),
            ("what rows are in live_check", "Cockpit lane: Acceptance Harness"),
            ("what rows does live_check show", "Cockpit lane: Acceptance Harness"),
            ("acceptance coverage status", "Cockpit lane: Acceptance Harness"),
            ("acceptance coverage drift", "Cockpit lane: Acceptance Harness"),
            ("acceptance gaps", "Cockpit lane: Acceptance Harness"),
            ("what's left to finish?", "Cockpit lane: Acceptance Harness"),
            ("what remains before Jarvis is done?", "Cockpit lane: Acceptance Harness"),
            ("remaining acceptance gaps", "Cockpit lane: Acceptance Harness"),
            ("acceptance next proof", "Jarvis completion next proof packet"),
            ("next acceptance proof", "Jarvis completion next proof packet"),
            ("next acceptance test", "Jarvis completion next proof packet"),
            ("next live proof", "Jarvis completion next proof packet"),
            ("next live test", "Jarvis completion next proof packet"),
            ("next proof to run", "Jarvis completion next proof packet"),
            ("what is the next acceptance proof?", "Jarvis completion next proof packet"),
            ("what proof is next?", "Jarvis completion next proof packet"),
            ("what proof should I run next?", "Jarvis completion next proof packet"),
            ("what should I prove next?", "Jarvis completion next proof packet"),
            ("what should I test next?", "Jarvis completion next proof packet"),
            ("what should operator test next?", "Jarvis completion next proof packet"),
            ("which proof is next?", "Jarvis completion next proof packet"),
            ("smoke status", "Cockpit lane: Acceptance Harness"),
            ("are tests green?", "Cockpit lane: Acceptance Harness"),
            ("aggregate smoke status", "Cockpit lane: Acceptance Harness"),
            ("reboot status", "Cockpit lane: Acceptance Harness"),
            ("will Jarvis survive reboot?", "Cockpit lane: Acceptance Harness"),
            ("daemon startup status", "Cockpit lane: Acceptance Harness"),
            ("voice proof status", "Cockpit lane: Acceptance Harness"),
            ("is Korean voice tested?", "Cockpit lane: Acceptance Harness"),
            ("telegram voice proof status", "Cockpit lane: Acceptance Harness"),
            ("voice proof matrix", "Pending voice-proof matrix"),
            ("voice proof result format", "Pending voice-proof matrix"),
            ("voice live proof format", "Pending voice-proof matrix"),
            ("telegram voice proof matrix", "Pending voice-proof matrix"),
            ("telegram voice proof result format", "Pending voice-proof matrix"),
            ("local voice proof matrix", "Pending voice-proof matrix"),
            ("local voice proof result format", "Pending voice-proof matrix"),
            ("korean voice proof matrix", "Pending voice-proof matrix"),
            ("korean voice proof result format", "Pending voice-proof matrix"),
            ("push to talk proof matrix", "Pending voice-proof matrix"),
            ("push-to-talk proof matrix", "Pending voice-proof matrix"),
            ("talk.py proof matrix", "Pending voice-proof matrix"),
            ("spoken reply proof matrix", "Pending voice-proof matrix"),
            ("voice speak proof matrix", "Pending voice-proof matrix"),
            ("what voice should I test?", "Pending voice-proof matrix"),
            ("what voice proofs should I test?", "Pending voice-proof matrix"),
            ("what local voice should I test?", "Pending voice-proof matrix"),
            ("what telegram voice should I test?", "Pending voice-proof matrix"),
            ("what Korean voice should I test?", "Pending voice-proof matrix"),
            ("how should I report voice proof?", "Pending voice-proof matrix"),
            ("how should I report local voice proof?", "Pending voice-proof matrix"),
            ("how should I report telegram voice proof?", "Pending voice-proof matrix"),
            ("how should I report Korean voice proof?", "Pending voice-proof matrix"),
            ("음성 증명 매트릭스", "Pending voice-proof matrix"),
            ("음성 증명 결과 형식", "Pending voice-proof matrix"),
            ("음성 뭐 테스트해?", "Pending voice-proof matrix"),
            ("음성 증명 뭐 해야 해?", "Pending voice-proof matrix"),
            ("음성 증명 어떻게 보고해?", "Pending voice-proof matrix"),
            ("텔레그램 음성 증명", "Pending voice-proof matrix"),
            ("로컬 음성 증명", "Pending voice-proof matrix"),
            ("한국어 음성 증명 형식", "Pending voice-proof matrix"),
            ("personal proofs lane", "Cockpit lane: Personal Proofs"),
            ("개인 증명 레인", "Cockpit lane: Personal Proofs"),
            ("personal proof matrix", "Pending personal-proof matrix"),
            ("personal proof result format", "Pending personal-proof matrix"),
            ("personal live proof format", "Pending personal-proof matrix"),
            ("what personal proofs should I test?", "Pending personal-proof matrix"),
            ("what personal integrations should I test?", "Pending personal-proof matrix"),
            ("how should I report personal proof results?", "Pending personal-proof matrix"),
            ("what calendar write should I test", "Pending personal-proof matrix"),
            ("how should I report calendar write proof", "Pending personal-proof matrix"),
            ("calendar create update delete proof matrix", "Pending personal-proof matrix"),
            ("calendar write result format", "Pending personal-proof matrix"),
            ("email read proof matrix", "Pending personal-proof matrix"),
            ("email search proof matrix", "Pending personal-proof matrix"),
            ("email send proof matrix", "Pending personal-proof matrix"),
            ("what email read should I test", "Pending personal-proof matrix"),
            ("what email search should I test", "Pending personal-proof matrix"),
            ("what email send should I test", "Pending personal-proof matrix"),
            ("how should I report email read proof", "Pending personal-proof matrix"),
            ("how should I report email search proof", "Pending personal-proof matrix"),
            ("how should I report email send proof", "Pending personal-proof matrix"),
            ("reminder proof matrix", "Pending personal-proof matrix"),
            ("set reminder proof matrix", "Pending personal-proof matrix"),
            ("what reminder proof should I test", "Pending personal-proof matrix"),
            ("what set reminder proof should I test", "Pending personal-proof matrix"),
            ("how should I report reminder proof", "Pending personal-proof matrix"),
            ("how should I report set reminder proof", "Pending personal-proof matrix"),
            ("contact lookup proof matrix", "Pending personal-proof matrix"),
            ("what contact lookup should I test", "Pending personal-proof matrix"),
            ("how should I report contact lookup proof", "Pending personal-proof matrix"),
            ("personal integration proof matrix", "Pending personal-proof matrix"),
            ("personal integration result format", "Pending personal-proof matrix"),
            ("personal integrations proof matrix", "Pending personal-proof matrix"),
            ("personal integrations result format", "Pending personal-proof matrix"),
            ("개인 증명 매트릭스", "Pending personal-proof matrix"),
            ("개인 증명 결과 형식", "Pending personal-proof matrix"),
            ("개인 증명 뭐 해야 해?", "Pending personal-proof matrix"),
            ("개인 연동 뭐 테스트해?", "Pending personal-proof matrix"),
            ("개인 증명 어떻게 보고해?", "Pending personal-proof matrix"),
            ("캘린더 쓰기 증명 매트릭스", "Pending personal-proof matrix"),
            ("이메일 읽기 증명", "Pending personal-proof matrix"),
            ("이메일 검색 증명", "Pending personal-proof matrix"),
            ("이메일 보내기 증명", "Pending personal-proof matrix"),
            ("리마인더 증명 매트릭스", "Pending personal-proof matrix"),
            ("연락처 조회 증명 매트릭스", "Pending personal-proof matrix"),
            ("scheduler proof matrix", "Pending scheduler-proof matrix"),
            ("scheduler proof result format", "Pending scheduler-proof matrix"),
            ("scheduled jobs proof matrix", "Pending scheduler-proof matrix"),
            ("morning brief delivery proof matrix", "Pending scheduler-proof matrix"),
            ("jobs 7 day proof", "Pending scheduler-proof matrix"),
            ("jobs seven day proof", "Pending scheduler-proof matrix"),
            ("scheduled jobs 7 day proof", "Pending scheduler-proof matrix"),
            ("scheduled jobs streak", "Pending scheduler-proof matrix"),
            ("scheduled job streak status", "Pending scheduler-proof matrix"),
            ("scheduler streak status", "Pending scheduler-proof matrix"),
            ("did scheduled jobs run for 7 days?", "Pending scheduler-proof matrix"),
            ("are scheduled jobs running daily?", "Pending scheduler-proof matrix"),
            ("what is the 7 day scheduler proof?", "Pending scheduler-proof matrix"),
            ("what scheduled jobs should I test?", "Pending scheduler-proof matrix"),
            ("what scheduler streak should I test?", "Pending scheduler-proof matrix"),
            ("how should I report scheduler proof?", "Pending scheduler-proof matrix"),
            ("how should I report scheduled job streak proof?", "Pending scheduler-proof matrix"),
            ("7-day scheduler proof", "Pending scheduler-proof matrix"),
            ("daily streak proof", "Pending scheduler-proof matrix"),
            ("daily job streak proof", "Pending scheduler-proof matrix"),
            ("job streak proof matrix", "Pending scheduler-proof matrix"),
            ("scheduled delivery proof matrix", "Pending scheduler-proof matrix"),
            ("스케줄 증명 매트릭스", "Pending scheduler-proof matrix"),
            ("스케줄 증명 결과 형식", "Pending scheduler-proof matrix"),
            ("스케줄 증명 뭐 해야 해?", "Pending scheduler-proof matrix"),
            ("예약 작업 뭐 테스트해?", "Pending scheduler-proof matrix"),
            ("예약 작업 증명 형식", "Pending scheduler-proof matrix"),
            ("approval proof matrix", "Pending approval-proof matrix"),
            ("approval proof result format", "Pending approval-proof matrix"),
            ("phone approval proof", "Pending approval-proof matrix"),
            ("approval buttons proof matrix", "Pending approval-proof matrix"),
            ("phone approval buttons proof matrix", "Pending approval-proof matrix"),
            ("approval callback proof matrix", "Pending approval-proof matrix"),
            ("what approval buttons should I test?", "Pending approval-proof matrix"),
            ("how should I report approval callback proof?", "Pending approval-proof matrix"),
            ("what approval flow should I test?", "Pending approval-proof matrix"),
            ("how should I report approval proof?", "Pending approval-proof matrix"),
            ("승인 증명 매트릭스", "Pending approval-proof matrix"),
            ("승인 결과 형식", "Pending approval-proof matrix"),
            ("승인 증명 뭐 해야 해?", "Pending approval-proof matrix"),
            ("승인 흐름 뭐 테스트해?", "Pending approval-proof matrix"),
            ("폰 승인 버튼 증명", "Pending approval-proof matrix"),
            ("텔레그램 승인 증명", "Pending approval-proof matrix"),
            ("승인 콜백 증명", "Pending approval-proof matrix"),
            ("승인 거절 증명", "Pending approval-proof matrix"),
            ("phone control proof matrix", "Pending phone-control proof matrix"),
            ("phone control proof result format", "Pending phone-control proof matrix"),
            ("phone control live proof format", "Pending phone-control proof matrix"),
            ("what broke proof matrix", "Pending phone-control proof matrix"),
            ("cockpit proof matrix", "Pending phone-control proof matrix"),
            ("cockpit approvals proof matrix", "Pending phone-control proof matrix"),
            ("what phone control should I test?", "Pending phone-control proof matrix"),
            ("what phone shortcuts should I test?", "Pending phone-control proof matrix"),
            ("how should I report phone control proof?", "Pending phone-control proof matrix"),
            ("how should I report phone shortcut proof?", "Pending phone-control proof matrix"),
            ("폰컨트롤 증명 매트릭스", "Pending phone-control proof matrix"),
            ("폰 제어 증명 형식", "Pending phone-control proof matrix"),
            ("뭐가 고장났어 증명 형식", "Pending phone-control proof matrix"),
            ("reboot proof matrix", "Pending reboot-proof matrix"),
            ("reboot proof result format", "Pending reboot-proof matrix"),
            ("daemon proof matrix", "Pending reboot-proof matrix"),
            ("daemon proof result format", "Pending reboot-proof matrix"),
            ("daemon startup proof matrix", "Pending reboot-proof matrix"),
            ("launchagent proof matrix", "Pending reboot-proof matrix"),
            ("launchd proof matrix", "Pending reboot-proof matrix"),
            ("post reboot proof matrix", "Pending reboot-proof matrix"),
            ("post-reboot proof matrix", "Pending reboot-proof matrix"),
            ("network recovery proof matrix", "Pending reboot-proof matrix"),
            ("network recovery proof result format", "Pending reboot-proof matrix"),
            ("network loss proof matrix", "Pending reboot-proof matrix"),
            ("network loss proof result format", "Pending reboot-proof matrix"),
            ("telegram control daemon status", "Pending reboot-proof matrix"),
            ("telegram control proof matrix", "Pending reboot-proof matrix"),
            ("status server proof matrix", "Pending reboot-proof matrix"),
            ("scheduler daemon proof matrix", "Pending reboot-proof matrix"),
            ("what reboot proof should I test?", "Pending reboot-proof matrix"),
            ("what daemon proof should I test?", "Pending reboot-proof matrix"),
            ("what network recovery should I test?", "Pending reboot-proof matrix"),
            ("what should I test after reboot?", "Pending reboot-proof matrix"),
            ("how should I report reboot proof?", "Pending reboot-proof matrix"),
            ("how should I report daemon proof?", "Pending reboot-proof matrix"),
            ("how should I report network recovery proof?", "Pending reboot-proof matrix"),
            ("재부팅 증명 매트릭스", "Pending reboot-proof matrix"),
            ("재부팅 증명 형식", "Pending reboot-proof matrix"),
            ("데몬 증명 매트릭스", "Pending reboot-proof matrix"),
            ("네트워크 복구 증명", "Pending reboot-proof matrix"),
            ("error guidance proof matrix", "Pending error-guidance proof matrix"),
            ("error guidance proof result format", "Pending error-guidance proof matrix"),
            ("error message proof matrix", "Pending error-guidance proof matrix"),
            ("recovery guidance proof matrix", "Pending error-guidance proof matrix"),
            ("what error messages should I test?", "Pending error-guidance proof matrix"),
            ("how should I report error guidance proof?", "Pending error-guidance proof matrix"),
            ("do error messages name the fix", "Pending error-guidance proof matrix"),
            ("do user errors name the fix", "Pending error-guidance proof matrix"),
            ("are error messages actionable", "Pending error-guidance proof matrix"),
            ("is error guidance proven", "Pending error-guidance proof matrix"),
            ("error messages status", "Pending error-guidance proof matrix"),
            ("error recovery status", "Pending error-guidance proof matrix"),
            ("error recovery proof", "Pending error-guidance proof matrix"),
            ("what error messages should I test", "Pending error-guidance proof matrix"),
            ("how should I report error messages", "Pending error-guidance proof matrix"),
            ("how should I report error recovery proof", "Pending error-guidance proof matrix"),
            ("what recovery guidance should I test", "Pending error-guidance proof matrix"),
            ("what user-facing errors need proof", "Pending error-guidance proof matrix"),
            ("user-visible error proof matrix", "Pending error-guidance proof matrix"),
            ("user visible error proof matrix", "Pending error-guidance proof matrix"),
            ("connector error proof matrix", "Pending error-guidance proof matrix"),
            ("connector recovery proof matrix", "Pending error-guidance proof matrix"),
            ("recovery guidance status", "Pending error-guidance proof matrix"),
            ("show recovery guidance", "Pending error-guidance proof matrix"),
            ("show error guidance", "Pending error-guidance proof matrix"),
            ("오류 안내 증명 매트릭스", "Pending error-guidance proof matrix"),
            ("오류 안내 증명", "Pending error-guidance proof matrix"),
            ("오류 안내 형식", "Pending error-guidance proof matrix"),
            ("오류 메시지 상태", "Pending error-guidance proof matrix"),
            ("오류 복구 상태", "Pending error-guidance proof matrix"),
            ("복구 안내 증명", "Pending error-guidance proof matrix"),
            ("에러 안내 증명", "Pending error-guidance proof matrix"),
            ("mixed conversation proof matrix", "Pending mixed-conversation proof matrix"),
            ("mixed conversation proof result format", "Pending mixed-conversation proof matrix"),
            ("conversation proof matrix", "Pending mixed-conversation proof matrix"),
            ("conversation proof result format", "Pending mixed-conversation proof matrix"),
            ("chat latency proof matrix", "Pending mixed-conversation proof matrix"),
            ("chat latency proof result format", "Pending mixed-conversation proof matrix"),
            ("what mixed conversation should I test?", "Pending mixed-conversation proof matrix"),
            ("what conversation should I test?", "Pending mixed-conversation proof matrix"),
            ("what chat proof should I run?", "Pending mixed-conversation proof matrix"),
            ("how should I report mixed conversation proof?", "Pending mixed-conversation proof matrix"),
            ("how should I report conversation proof?", "Pending mixed-conversation proof matrix"),
            ("how should I report chat latency proof?", "Pending mixed-conversation proof matrix"),
            ("대화 증명 매트릭스", "Pending mixed-conversation proof matrix"),
            ("대화 증명 결과 형식", "Pending mixed-conversation proof matrix"),
            ("대화 증명 뭐 해야 해?", "Pending mixed-conversation proof matrix"),
            ("대화 증명 어떻게 보고해?", "Pending mixed-conversation proof matrix"),
            ("혼합 대화 증명", "Pending mixed-conversation proof matrix"),
            ("혼합 대화 뭐 테스트해?", "Pending mixed-conversation proof matrix"),
            ("research lane", "Cockpit lane: Research & Web"),
            ("web lane", "Cockpit lane: Research & Web"),
            ("검색 레인", "Cockpit lane: Research & Web"),
            ("연구 레인", "Cockpit lane: Research & Web"),
            ("what is frozen", "Capability cockpit"),
            ("what can Codex touch?", "Capability cockpit"),
            ("what should you not edit?", "Capability cockpit"),
            ("proofs pending", "Capability cockpit"),
            ("show guardrails", "Capability cockpit"),
            ("what is the live test matrix?", "Pending live-proof matrix"),
            ("why frozen?", "Capability cockpit"),
            ("safe lane", "Capability cockpit"),
            ("what should I test?", "Pending live-proof matrix"),
            ("test matrix status", "Pending live-proof matrix"),
            ("what live proofs are pending?", "Pending live-proof matrix"),
            ("what live tests are pending?", "Pending live-proof matrix"),
            ("what channels should I test?", "Pending live-proof matrix"),
            ("which channels need live proof?", "Pending live-proof matrix"),
            ("what live channels are pending?", "Pending live-proof matrix"),
            ("what results do you need from me?", "Pending live-proof matrix"),
            ("what should operator test?", "Pending live-proof matrix"),
            ("what tests should I run?", "Pending live-proof matrix"),
            ("how should I report live test results?", "Pending live-proof matrix"),
            ("how do I report live test results?", "Pending live-proof matrix"),
            ("live test result format", "Pending live-proof matrix"),
            ("what format should I use for live test results?", "Pending live-proof matrix"),
            ("what should I send after testing?", "Pending live-proof matrix"),
            ("report live matrix results", "Pending live-proof matrix"),
            ("report live test results", "Pending live-proof matrix"),
            ("프리즈 상태", "Capability cockpit"),
            ("동결 파일", "Capability cockpit"),
            ("검증 대기 뭐야?", "Pending live-proof matrix"),
            ("테스트 매트릭스 보여줘", "Pending live-proof matrix"),
            ("라이브 테스트 뭐 해야 해?", "Pending live-proof matrix"),
            ("어떤 채널 테스트해?", "Pending live-proof matrix"),
            ("라이브 채널 뭐 테스트해?", "Pending live-proof matrix"),
            ("테스트 매트릭스 상태", "Pending live-proof matrix"),
            ("라이브 증명 상태", "Cockpit lane: Acceptance Harness"),
            ("인수 상태", "Cockpit lane: Acceptance Harness"),
            ("남은 기준", "Cockpit lane: Acceptance Harness"),
            ("뭐 남았어?", "Cockpit lane: Acceptance Harness"),
            ("다음 증명", "Jarvis completion next proof packet"),
            ("다음 라이브 테스트", "Jarvis completion next proof packet"),
            ("다음에 뭐 테스트해", "Jarvis completion next proof packet"),
            ("다음에 뭐 증명해", "Jarvis completion next proof packet"),
            ("라이브체크 커버리지", "Cockpit lane: Acceptance Harness"),
            ("라이브 체크 표", "Cockpit lane: Acceptance Harness"),
            ("라이브체크 항목", "Cockpit lane: Acceptance Harness"),
            ("원스크린 인수 표", "Cockpit lane: Acceptance Harness"),
            ("인수 커버리지", "Cockpit lane: Acceptance Harness"),
            ("뭐 증명해야 해?", "Cockpit lane: Acceptance Harness"),
            ("스모크 상태", "Cockpit lane: Acceptance Harness"),
            ("테스트 초록", "Cockpit lane: Acceptance Harness"),
            ("재부팅 상태", "Cockpit lane: Acceptance Harness"),
            ("데몬 상태", "Cockpit lane: Acceptance Harness"),
            ("음성 증명 상태", "Cockpit lane: Acceptance Harness"),
            ("한국어 음성 증명", "Cockpit lane: Acceptance Harness"),
            ("뭐 테스트해야 해?", "Pending live-proof matrix"),
            ("무슨 테스트해야 해?", "Pending live-proof matrix"),
            ("어떤 테스트해야 해?", "Pending live-proof matrix"),
            ("테스트해야 할 거", "Pending live-proof matrix"),
            ("라이브 테스트 결과 보고", "Pending live-proof matrix"),
            ("라이브 결과 보고 방법", "Pending live-proof matrix"),
            ("라이브 결과 형식", "Pending live-proof matrix"),
            ("라이브 테스트 결과 형식", "Pending live-proof matrix"),
            ("테스트 결과 어떻게 보고해?", "Pending live-proof matrix"),
            ("테스트 결과 어떻게 보내?", "Pending live-proof matrix"),
            ("테스트 결과 형식", "Pending live-proof matrix"),
            ("수정 금지 파일", "Capability cockpit"),
            ("동결 상태", "Capability cockpit"),
        )
        for phrase, expected in direct_cases:
            before = len(runtime.inputs)
            reply, markup = bridge._handle_command(phrase)
            if markup is not None:
                return False, "phone control cockpit shortcut unexpectedly returned approval buttons"
            if expected == "Jarvis completion next proof packet":
                if len(runtime.inputs) != before + 1 or runtime.inputs[-1] != "completion_next_proof_packet":
                    return False, "phone control next-proof shortcuts should route only to the read-only next-proof packet"
            elif len(runtime.inputs) != before:
                return False, "phone control cockpit shortcuts should not route through runtime.handle"
            if expected not in reply:
                return False, "phone control cockpit shortcut lost its summary reply"
            if "read-only" not in reply:
                return False, "phone control cockpit shortcut lost its read-only boundary"
            if expected == "Capability cockpit" and (
                "Build Guardrails" not in reply
                or "Acceptance Harness" not in reply
                or "live_check publishes the one-screen acceptance table" not in reply
                or "readiness rows do not replace live delivery" not in reply
                or "live-proof freeze" not in reply
                or "planner/send/call/HUD" not in reply
                or "pending live-proof matrix covers Kakao" not in reply
            ):
                return False, "phone control freeze shortcut lost live-proof guardrail context"
            if expected == "Pending live-proof matrix" and (
                "KakaoTalk" not in reply
                or "Telegram" not in reply
                or "Instagram" not in reply
                or "iMessage" not in reply
                or "Phone:" not in reply
                or "FaceTime:" not in reply
                or "KakaoTalk call:" not in reply
                or "Telegram call:" not in reply
                or "Instagram call:" not in reply
                or "approval card's channel, target, and effective mode match" not in reply
                or "call_requested / known_not_started / outcome_unknown" not in reply
                or "recipient confirmation" not in reply
                or "Report format" not in reply
                or "pass / fail / blocked" not in reply
                or "last visible stage/error" not in reply
                or "do not include secrets" not in reply
                or "message content" not in reply
                or "does not approve, send, call, run live_check" not in reply
            ):
                return False, "phone control live-test shortcut lost compact matrix/reporting context"
            if expected == "Pending live-proof matrix" and any(
                target in reply for target in ("Fixture", "가상연락처일", "가상연락처이", "BotFather")
            ):
                return False, "phone control live-test shortcut echoed private target names"
            if expected == "Pending personal-proof matrix" and (
                "Calendar write cycle" not in reply
                or "Email read/search" not in reply
                or "Email send" not in reply
                or "Reminder delivery" not in reply
                or "Contacts lookup" not in reply
                or "Report format" not in reply
                or "area: calendar / email-read / email-send / reminder / contacts" not in reply
                or "result: pass / fail / blocked" not in reply
                or "approval card shown: yes / no / n/a" not in reply
                or "last visible stage/error" not in reply
                or "Privacy boundary" not in reply
                or "does not access accounts, read email" not in reply
                or "claim personal proof done" not in reply
            ):
                return False, "phone control personal-proof shortcut lost compact matrix/reporting context"
            if expected == "Pending personal-proof matrix" and any(
                target in reply
                for target in (
                    "Fixture",
                    "가상연락처일",
                    "가상연락처이",
                    "@",
                    "synthetic_owner_handle",
                    "message body",
                    "email body",
                )
            ):
                return False, "phone control personal-proof shortcut leaked private target/report details"
            if expected == "Pending scheduler-proof matrix" and (
                "Morning Brief delivery" not in reply
                or "Scheduler freshness" not in reply
                or "7-day streak" not in reply
                or "Silent-failure check" not in reply
                or "Recovery check" not in reply
                or "Report format" not in reply
                or "area: morning-brief / freshness / 7-day-streak / silent-failure / recovery" not in reply
                or "result: pass / fail / blocked" not in reply
                or "date range: YYYY-MM-DD..YYYY-MM-DD" not in reply
                or "enabled jobs: count only" not in reply
                or "last visible stage/error" not in reply
                or "Privacy boundary" not in reply
                or "does not run jobs, send Telegram messages, restart daemons" not in reply
                or "claim scheduler proof done" not in reply
            ):
                return False, "phone control scheduler-proof shortcut lost compact matrix/reporting context"
            if expected == "Pending scheduler-proof matrix" and any(
                target in reply
                for target in (
                    "/\x55sers/",
                    "/private/",
                    "Fixture",
                    "가상연락처일",
                    "가상연락처이",
                    "synthetic_owner_handle",
                )
            ):
                return False, "phone control scheduler-proof shortcut leaked private target/report details"
            if expected == "Pending approval-proof matrix" and (
                "Risky action prompt" not in reply
                or "Approval last-look" not in reply
                or "Owner approve path" not in reply
                or "Owner deny path" not in reply
                or "Non-owner guard" not in reply
                or "Recovery check" not in reply
                or "Report format" not in reply
                or "area: prompt / last-look / approve / deny / non-owner / recovery" not in reply
                or "result: pass / fail / blocked" not in reply
                or "approval id: number only, or n/a" not in reply
                or "callback shown: approve / deny / both / none" not in reply
                or "last visible stage/error" not in reply
                or "Privacy boundary" not in reply
                or "does not create approvals, approve, dismiss, execute tools" not in reply
                or "claim approval proof done" not in reply
            ):
                return False, "phone control approval-proof shortcut lost compact matrix/reporting context"
            if expected == "Pending approval-proof matrix" and any(
                target in reply
                for target in (
                    "/\x55sers/",
                    "/private/",
                    "Fixture",
                    "가상연락처일",
                    "가상연락처이",
                    "synthetic_owner_handle",
                    "approve:7",
                    "deny:7",
                    "555001",
                    "999999",
                )
            ):
                return False, "phone control approval-proof shortcut leaked private target/report details"
            if expected == "Pending phone-control proof matrix" and (
                "What broke" not in reply
                or "Cockpit summary" not in reply
                or "Cockpit attention" not in reply
                or "Approvals lane" not in reply
                or "Owner-only guard" not in reply
                or "Recovery check" not in reply
                or "Report format" not in reply
                or "area: what-broke / cockpit-summary / attention / approvals / owner-guard / recovery" not in reply
                or "result: pass / fail / blocked" not in reply
                or "approval-held shown: yes / no / n/a" not in reply
                or "failure lane shown: yes / no / n/a" not in reply
                or "last visible stage/error" not in reply
                or "optional live_check row: phone control / channel health / phone approvals" not in reply
                or "Privacy boundary" not in reply
                or "do not include message contents" not in reply
                or "does not inspect private payloads, create approvals, approve, dismiss, execute tools" not in reply
                or "claim phone-control proof done" not in reply
            ):
                return False, "phone control proof shortcut lost compact matrix/reporting context"
            if expected == "Pending phone-control proof matrix" and any(
                target in reply
                for target in (
                    "/\x55sers/",
                    "/private/",
                    "Fixture",
                    "가상연락처일",
                    "가상연락처이",
                    "synthetic_owner_handle",
                    "full transcript",
                    "planned args:",
                    "run payload:",
                    "chat_id",
                    "approve:7",
                    "deny:7",
                    "555001",
                    "999999",
                )
            ):
                return False, "phone control proof shortcut leaked private target/report details"
            if expected == "Pending reboot-proof matrix" and (
                "Post-reboot status" not in reply
                or "LaunchAgent install" not in reply
                or "Scheduler freshness" not in reply
                or "Telegram control" not in reply
                or "Network recovery" not in reply
                or "Recovery check" not in reply
                or "Report format" not in reply
                or "area: post-reboot / launchagent / scheduler / telegram-control / network-recovery / recovery" not in reply
                or "result: pass / fail / blocked" not in reply
                or "daemon: telegram-control / scheduler / status-server / all / n/a" not in reply
                or "reboot observed: yes / no / n/a" not in reply
                or "network-loss observed: yes / no / n/a" not in reply
                or "last visible stage/error" not in reply
                or "optional live_check row: daemon startup / scheduled jobs / phone control / channel health" not in reply
                or "Privacy boundary" not in reply
                or "do not include plist paths" not in reply
                or "does not run launchctl, install or edit LaunchAgents, restart daemons" not in reply
                or "claim reboot proof done" not in reply
            ):
                return False, "reboot proof shortcut lost compact matrix/reporting context"
            if expected == "Pending reboot-proof matrix" and any(
                target in reply
                for target in (
                    "/\x55sers/",
                    "/private/",
                    "Fixture",
                    "가상연락처일",
                    "가상연락처이",
                    "synthetic_owner_handle",
                    "chat_id",
                    "launchctl load",
                    "launchctl unload",
                    "launchctl bootout",
                    "approve:7",
                    "deny:7",
                    "555001",
                    "999999",
                )
            ):
                return False, "reboot proof shortcut leaked private target/report details"
            if expected == "Pending error-guidance proof matrix" and (
                "Calendar/Gmail setup" not in reply
                or "Reminders" not in reply
                or "Voice/Whisper" not in reply
                or "Model/Ollama" not in reply
                or "Generic connector recovery" not in reply
                or "Recovery check" not in reply
                or "Report format" not in reply
                or "area: calendar / gmail / reminders / voice / model / connector / recovery" not in reply
                or "result: pass / fail / blocked" not in reply
                or "fix named: yes / no / wrong" not in reply
                or "raw internals leaked: yes / no" not in reply
                or "last visible stage/error" not in reply
                or "optional live_check row: error guidance" not in reply
                or "Privacy boundary" not in reply
                or "do not include email subjects" not in reply
                or "does not access accounts, trigger failing operations, call models" not in reply
                or "claim error-guidance proof done" not in reply
            ):
                return False, "error guidance proof shortcut lost compact matrix/reporting context"
            if expected == "Pending error-guidance proof matrix" and any(
                target in reply
                for target in (
                    "/\x55sers/",
                    "/private/",
                    "Fixture",
                    "가상연락처일",
                    "가상연락처이",
                    "synthetic_owner_handle",
                    "Traceback",
                    "chat_id",
                    "approve:7",
                    "deny:7",
                    "555001",
                    "999999",
                )
            ):
                return False, "error guidance proof shortcut leaked private target/report details"
            if expected == "Pending mixed-conversation proof matrix" and (
                "10+ turn run" not in reply
                or "Coherence check" not in reply
                or "Latency check" not in reply
                or "chat p95 is <= 8000ms" not in reply
                or "Model path check" not in reply
                or "Recovery check" not in reply
                or "Report format" not in reply
                or "area: turn-mix / coherence / latency / model-path / recovery" not in reply
                or "result: pass / fail / blocked" not in reply
                or "turn count: number only" not in reply
                or "chat p95 ms: number only, or n/a" not in reply
                or "slowest stage/error" not in reply
                or "optional live_check row: mixed conversation" not in reply
                or "Privacy boundary" not in reply
                or "do not include transcript contents" not in reply
                or "does not run a conversation, call the model, change tuning knobs" not in reply
                or "claim the mixed-conversation proof done" not in reply
            ):
                return False, "phone control mixed-conversation proof shortcut lost compact matrix/reporting context"
            if expected == "Pending mixed-conversation proof matrix" and any(
                target in reply
                for target in (
                    "/\x55sers/",
                    "/private/",
                    "Fixture",
                    "가상연락처일",
                    "가상연락처이",
                    "synthetic_owner_handle",
                    "full transcript",
                    "calendar title",
                    "task body",
                )
            ):
                return False, "phone control mixed-conversation proof shortcut leaked private target/report details"
            if expected == "Pending voice-proof matrix" and (
                "Telegram voice note" not in reply
                or "Local push-to-talk" not in reply
                or "Spoken reply" not in reply
                or "Korean voice" not in reply
                or "Recovery check" not in reply
                or "Report format" not in reply
                or "area: telegram-voice / local-talk / speak / korean-transcription / recovery" not in reply
                or "result: pass / fail / blocked" not in reply
                or "language: en / ko / mixed / n/a" not in reply
                or "approval card shown: yes / no / n/a" not in reply
                or "last visible stage/error" not in reply
                or "optional live_check row: telegram voice / local talk / voice setup" not in reply
                or "Privacy boundary" not in reply
                or "do not include transcript contents" not in reply
                or "does not access the microphone, record audio, download voice files" not in reply
                or "claim the voice proof done" not in reply
            ):
                return False, "phone control voice-proof shortcut lost compact matrix/reporting context"
            if expected == "Pending voice-proof matrix" and any(
                target in reply
                for target in (
                    "/\x55sers/",
                    "/private/",
                    "Fixture",
                    "가상연락처일",
                    "가상연락처이",
                    "synthetic_owner_handle",
                    "full transcript",
                    "voice_file_id",
                    "file_id",
                    "chat_id",
                    "555001",
                    "999999",
                )
            ):
                return False, "phone control voice-proof shortcut leaked private target/report details"
            if expected == "Cockpit lane: Acceptance Harness" and (
                "READ_ONLY" not in reply
                or "approval required: no" not in reply
                or "live proof status" not in reply
                or "acceptance gaps" not in reply
                or "acceptance next" not in reply
                or "acceptance gaps points to acceptance next for prioritized proof guidance" not in reply
                or "aggregate smoke" not in reply
                or "live_check publishes the one-screen acceptance table" not in reply
                or "diagnostic only; run opt-in live_check with operator present" not in reply
                or "readiness rows do not replace live delivery" not in reply
                or "This is read-only" not in reply
            ):
                return False, "phone control acceptance shortcut lost compact live-proof context"
            if expected == "Cockpit lane: Personal Proofs" and (
                "HIGH_RISK" not in reply
                or "approval required: yes" not in reply
                or "personal proofs status" not in reply
                or "personal proofs are still live-acceptance evidence" not in reply
                or "set_reminder proof uses Jarvis's local Telegram reminder path" not in reply
                or "does not access accounts or send anything" not in reply
                or "This is read-only" not in reply
            ):
                return False, "phone control personal proofs shortcut lost WS2 proof boundary"
            if expected == "Cockpit lane: Research & Web" and (
                "LOCAL_SAFE" not in reply
                or "approval required: no" not in reply
                or "research Zoey OS" not in reply
                or "research and web_lookup emit no-authority handoff packets" not in reply
                or "does not fetch pages, synthesize answers, write notes, or queue approvals" not in reply
                or "This is read-only" not in reply
            ):
                return False, "phone control research/web shortcut lost no-authority boundary"
            if expected == "Cockpit summary" and "approval-held" not in reply:
                return False, "phone control summary shortcut lost approval-held count"
            if expected == "Cockpit summary" and "not counted as a tool failure" not in reply:
                return False, "phone control summary shortcut lost approval-held safety semantics"
            if expected == "Cockpit summary" and "trust checklist 4/4 lanes, 12/12 checks ready" not in reply:
                return False, "phone control summary shortcut lost earned-trust checklist context"
            if expected == "Cockpit summary" and (
                "2 more proof lane(s) available" not in reply
                or "Learning Loop" not in reply
                or "Internal Orchestration" not in reply
                or "ask for `<lane> lane`" not in reply
            ):
                return False, "phone control summary shortcut lost hidden proof-lane discovery"
            if expected == "Cockpit attention" and (
                "Needs attention:" not in reply
                or "Approval-held review:" not in reply
                or "approval readiness 7" not in reply
            ):
                return False, "phone control attention shortcut lost approval-held split"
            if expected == "Cockpit attention" and "not counted as a tool failure" not in reply:
                return False, "phone control attention shortcut lost approval-held safety semantics"
            if expected == "Cockpit lane: Approvals" and "not counted as a tool failure" not in reply:
                return False, "phone control approvals lane lost approval-held safety semantics"
            if phrase in {"worker status", "워커 상태"} and (
                "READ_ONLY" not in reply or "internal subagents are workers" not in reply
            ):
                return False, "phone control worker lane lost internal-worker boundary"
            if expected == "Cockpit proofs:" and (
                "coverage: 3 lane(s), 9 proof point(s)" not in reply
                or "Korean Telegram send stops at approval" not in reply
                or "Morning Brief can be pushed" not in reply
                or "2 more proof lane(s) available" not in reply
                or "Learning Loop" not in reply
                or "Internal Orchestration" not in reply
            ):
                return False, "phone control proof shortcut lost compact proof context"
            if "Operator Workflow Evals" in expected and (
                "HIGH_RISK" not in reply or "approval required: yes" not in reply or "smoke: registered" not in reply
            ):
                return False, "phone control eval shortcut lost trust/eval lane context"
    except Exception as e:
        return False, f"EXCEPTION: {_safe_exception(e)}"
    return True, (
        "ready; phone control shortcuts route status, voice, capabilities, agent research/Zoey, "
        "safety, privacy, risk, setup, doctor, brief, channels, approvals, memory/state, "
        "learning loop, progress/roadmap/goals, AGI gates, worker readiness/internal workers, "
        "cockpit, completion-blocker proof shortcuts, live-test matrix/report-format packet, personal-proof matrix/report-format packet, scheduler-proof matrix/report-format packet, approval-proof matrix/report-format packet, phone-control proof matrix/report-format packet, reboot-proof matrix/report-format packet, error-guidance proof matrix/report-format packet, mixed-conversation proof matrix/report-format packet, voice-proof matrix/report-format packet, acceptance harness/live proof status, acceptance-gaps proof shortcuts, acceptance-next proof shortcuts, aggregate-smoke proof shortcuts, reboot/daemon proof shortcuts, voice-proof shortcuts, personal proofs lane, research/web lane, compact proof view, hidden proof-lane discovery, delivery-status/did-it-send/call-connected, what-can-you-do discovery, "
        "trust checklist, help discovery, guardrails/freeze, approval-held attention split, "
        "and trust/eval surfaces read-only"
    )


def _check_morning_brief_schedule(config) -> tuple[bool | None, str]:
    """Read-only: is the scheduled Morning Brief job set up, and when does it next run?"""
    try:
        from jarvis_v2.automations.scheduler import MORNING_BRIEF_JOB_NAME
        from jarvis_v2.memory.store import MemoryStore

        store = MemoryStore(config.db_path)
        jobs = store.list_jobs()
    except Exception as e:
        return False, f"scheduler store unreadable: {_safe_exception(e)}"
    job = next(
        (
            j
            for j in (jobs or [])
            if _plain_text(_job_value(j, "name")).casefold() == MORNING_BRIEF_JOB_NAME.casefold()
        ),
        None,
    )
    if job is None:
        return None, "not scheduled — set JARVIS_MORNING_BRIEF=09:00 or 'schedule morning brief at 9:00am'"
    next_run = _safe_iso_datetime(_job_value(job, "next_run_at"))
    if not _job_enabled(job):
        return None, f"job exists but is paused — next would be {next_run}"
    if next_run == "invalid_timestamp":
        return False, "job exists but next_run_at is invalid_timestamp"
    return True, f"scheduled, next run {next_run}"


def _check_scheduled_job_freshness(config, now: datetime | None = None) -> tuple[bool | None, str]:
    """Read-only: are enabled scheduled jobs recently run or still waiting for first due time?"""
    try:
        from jarvis_v2.memory.store import MemoryStore

        store = MemoryStore(config.db_path)
        jobs = store.list_jobs()
    except Exception as e:
        return False, f"scheduler store unreadable: {_safe_exception(e)}"

    rows = list(jobs or [])
    if not rows:
        return None, "no scheduled jobs — run schedule assistant basics if the operator wants background upkeep"

    enabled_rows = [row for row in rows if _job_enabled(row)]
    if not enabled_rows:
        return None, f"0/{len(rows)} job(s) enabled — scheduled upkeep is paused"

    now_dt = (now or datetime.now()).replace(microsecond=0)
    invalid = 0
    overdue = 0
    stale_last_run = 0
    stale_jobs = 0
    never_run = 0
    next_due_values: list[datetime] = []

    for row in enabled_rows:
        interval = _safe_positive_int(_job_value(row, "interval_minutes"))
        next_run = _parse_safe_datetime(_job_value(row, "next_run_at"))
        last_text = _plain_text(_job_value(row, "last_run_at")).strip()
        last_run = _parse_safe_datetime(last_text) if last_text else None
        if interval is None or next_run is None or (last_text and last_run is None):
            invalid += 1
            continue

        next_due_values.append(next_run)
        is_overdue = next_run <= now_dt - timedelta(minutes=JOB_FRESHNESS_GRACE_MINUTES)
        if is_overdue:
            overdue += 1
        if last_run is None:
            never_run += 1
            if is_overdue:
                stale_jobs += 1
            continue
        is_stale_last_run = now_dt - last_run > timedelta(minutes=interval + JOB_FRESHNESS_GRACE_MINUTES)
        if is_stale_last_run:
            stale_last_run += 1
        if is_overdue or is_stale_last_run:
            stale_jobs += 1

    if invalid:
        return False, f"{invalid} enabled job(s) have invalid freshness metadata — inspect list scheduled jobs"

    if stale_jobs:
        return (
            False,
            (
                f"{len(enabled_rows)} enabled job(s); {stale_jobs} stale "
                f"({overdue} overdue next_run, {stale_last_run} old last_run) — "
                "check the scheduler daemon or run due jobs with operator present"
            ),
        )

    next_due = min(next_due_values).isoformat() if next_due_values else "unknown"
    if never_run:
        return (
            None,
            (
                f"{len(enabled_rows)} enabled job(s); {never_run} awaiting first recorded run; "
                f"no overdue jobs; next due {next_due}"
            ),
        )
    return (
        True,
        (
            f"{len(enabled_rows)} enabled job(s) fresh; last_run drift within interval+"
            f"{JOB_FRESHNESS_GRACE_MINUTES}m; next due {next_due}"
        ),
    )


def _job_history_date(value: object) -> date | None:
    try:
        return date.fromisoformat(_plain_text(value).strip())
    except (TypeError, ValueError):
        return None


def _job_history_events(
    row: object,
) -> tuple[list[tuple[datetime, date, str, str, str, int]], int]:
    metadata = _job_metadata(row)
    raw_history = metadata.get(SCHEDULED_JOB_RUN_HISTORY_KEY)
    if raw_history in (None, "", []):
        return [], 0
    if not isinstance(raw_history, list):
        return [], 1

    if len(raw_history) > SCHEDULED_JOB_RUN_HISTORY_LIMIT:
        return [], 1
    schema = metadata.get("run_history_schema")
    if schema != 3:
        return [], 1
    events: list[tuple[datetime, date, str, str, str, int]] = []
    invalid = 0
    seen: set[str] = set()
    for raw_event in raw_history:
        if not isinstance(raw_event, dict):
            invalid += 1
            continue
        event_date = _job_history_date(raw_event.get("date"))
        ran_at_text = _plain_text(raw_event.get("ran_at")).strip()
        ran_at = _parse_safe_datetime(ran_at_text)
        try:
            encoded_ran_at = datetime.fromisoformat(ran_at_text)
        except (TypeError, ValueError):
            encoded_ran_at = None
        status = _plain_text(raw_event.get("status")).strip().casefold()
        occurrence_key = _plain_text(raw_event.get("occurrence_key")).strip()
        job_type = _plain_text(raw_event.get("job_type")).strip()
        schedule_identity_revision = raw_event.get("schedule_identity_revision")
        key_valid = bool(SCHEDULED_JOB_RUN_OCCURRENCE_RE.fullmatch(occurrence_key))
        if (
            event_date is None
            or ran_at is None
            or encoded_ran_at is None
            or encoded_ran_at.date() != event_date
            or status not in {"pending", "ok", "skipped", "failed"}
            or not key_valid
            or occurrence_key in seen
            or SCHEDULED_JOB_TYPE_RE.fullmatch(job_type) is None
            or type(schedule_identity_revision) is not int
            or (
                not (job_type == "legacy_unknown" and schedule_identity_revision == -1)
                and (
                    schedule_identity_revision < 0
                    or schedule_identity_revision > 9223372036854775807
                )
            )
        ):
            invalid += 1
            continue
        seen.add(occurrence_key)
        events.append(
            (
                ran_at,
                event_date,
                status,
                occurrence_key,
                job_type,
                schedule_identity_revision,
            )
        )
    return events, invalid


def _consecutive_dates_ending_at(days: set[date], end: date) -> int:
    streak = 0
    cursor = end
    while cursor in days:
        streak += 1
        cursor -= timedelta(days=1)
    return streak


def _scheduled_delivery_truth(
    rows: object,
    *,
    job_row: object,
    job_type: str,
    occurrence_key: str,
) -> str:
    """Return fail-closed terminal truth for one complete delivery chunk set."""
    if not isinstance(rows, (list, tuple)) or not rows:
        return "invalid"
    expected_job_id = _job_value(job_row, "id")
    if (
        type(expected_job_id) is not int
        or expected_job_id < 1
        or expected_job_id > 9223372036854775807
    ):
        return "invalid"
    chunk_count: int | None = None
    indexes: set[int] = set()
    states: set[str] = set()
    source_job_id: int | None = None
    receipt_job_id: int | None = None
    receipt_job_id_set = False
    for row in rows:
        index = _job_value(row, "chunk_index")
        count = _job_value(row, "chunk_count")
        state = _plain_text(_job_value(row, "state")).strip()
        row_job_id = _job_value(row, "job_id")
        row_source_job_id = _job_value(row, "source_job_id")
        is_morning_receipt = (
            job_type == "morning_brief_telegram" and occurrence_key.startswith("morning:")
        )
        if (
            type(index) is not int
            or type(count) is not int
            or index < 0
            or count < 1
            or index >= count
            or count > 32
            or (chunk_count is not None and count != chunk_count)
            or index in indexes
            or state
            not in {"prepared", "claimed", "sending", "accepted", "rejected", "uncertain", "failed"}
            or type(row_source_job_id) is not int
            or row_source_job_id < 1
            or (source_job_id is not None and row_source_job_id != source_job_id)
            or (receipt_job_id_set and row_job_id != receipt_job_id)
            or _plain_text(_job_value(row, "job_type")).strip() != job_type
        ):
            return "invalid"
        if is_morning_receipt:
            if row_job_id is not None and (
                type(row_job_id) is not int or row_job_id < 1
            ):
                return "invalid"
        elif row_job_id != expected_job_id or row_source_job_id != expected_job_id:
            return "invalid"
        chunk_count = count
        source_job_id = row_source_job_id
        receipt_job_id = row_job_id
        receipt_job_id_set = True
        indexes.add(index)
        states.add(state)
    if chunk_count is None or len(rows) != chunk_count or indexes != set(range(chunk_count)):
        return "invalid"
    if states & {"uncertain", "rejected", "failed"}:
        return "failed"
    if states == {"accepted"}:
        return "ok"
    return "pending"


def _check_scheduled_job_streak(config, now: datetime | None = None) -> tuple[bool | None, str]:
    """Read-only: is there enough per-job history to prove the 7-day scheduler DoD?"""
    try:
        from jarvis_v2.memory.store import MemoryStore

        store = MemoryStore(config.db_path)
        jobs = store.list_jobs()
    except Exception as e:
        return False, f"scheduler store unreadable: {_safe_exception(e)}"

    rows = list(jobs or [])
    if not rows:
        return None, "no scheduled jobs — run schedule assistant basics before collecting 7-day proof"

    enabled_rows = [row for row in rows if _job_enabled(row)]
    if len(enabled_rows) < SCHEDULED_JOB_EXPECTED_ENABLED_COUNT:
        return (
            None,
            (
                f"{len(enabled_rows)}/{SCHEDULED_JOB_EXPECTED_ENABLED_COUNT} expected enabled job(s); "
                "schedule Morning Brief and assistant basics before 7-day proof"
            ),
        )

    proof_rows: list[object] = []
    inventory_drift = 0
    for expected_type, expected_interval in SCHEDULED_JOB_EXPECTED_INTERVALS.items():
        candidates = [
            row
            for row in rows
            if _plain_text(_job_value(row, "job_type")).strip() == expected_type
        ]
        exact = [
            row
            for row in candidates
            if _job_enabled(row)
            and _safe_positive_int(_job_value(row, "interval_minutes")) == expected_interval
        ]
        if len(candidates) > 1 or len(exact) > 1:
            return False, "scheduled-job proof inventory has duplicate expected type(s)"
        if len(exact) != 1:
            inventory_drift += 1
            continue
        proof_rows.append(exact[0])
    if inventory_drift:
        return (
            None,
            (
                f"{len(proof_rows)}/{SCHEDULED_JOB_EXPECTED_ENABLED_COUNT} expected enabled job(s); "
                f"{inventory_drift} expected schedule definition(s) missing or changed"
            ),
        )
    enabled_rows = proof_rows
    enabled_count = len(enabled_rows)

    invalid_events = 0
    failed_events = 0
    pending_events = 0
    jobs_without_history = 0
    cadence_gaps = 0
    coverage_days: list[int] = []
    now_dt = now or datetime.now()
    if now_dt.tzinfo is not None:
        now_dt = now_dt.astimezone().replace(tzinfo=None)
    now_dt = now_dt.replace(microsecond=0)
    today = now_dt.date()
    proof_window = timedelta(days=SCHEDULED_JOB_STREAK_DAYS)
    proof_start = now_dt - proof_window
    grace = timedelta(minutes=JOB_FRESHNESS_GRACE_MINUTES)

    for row in enabled_rows:
        events, invalid = _job_history_events(row)
        invalid_events += invalid
        interval = _safe_positive_int(_job_value(row, "interval_minutes"))
        current_job_type = _plain_text(_job_value(row, "job_type")).strip()
        current_identity_revision = _job_value(row, "schedule_identity_revision")
        if (
            interval is None
            or SCHEDULED_JOB_TYPE_RE.fullmatch(current_job_type) is None
            or type(current_identity_revision) is not int
            or current_identity_revision < 0
            or current_identity_revision > 9223372036854775807
        ):
            invalid_events += 1
            continue
        if not events:
            jobs_without_history += 1
            coverage_days.append(0)
            continue

        allowed_gap = timedelta(minutes=interval) + grace
        matching_events = [
            event
            for event in events
            if event[4] == current_job_type and event[5] == current_identity_revision
        ]
        if not matching_events:
            jobs_without_history += 1
            coverage_days.append(0)
            continue

        window_events = [
            event
            for event in matching_events
            if event[0] >= proof_start - allowed_gap
        ]
        good_runs: list[datetime] = []
        for ran_at, event_date, status, occurrence_key, _job_type, _revision in window_events:
            if event_date > today or ran_at > now_dt + grace:
                invalid_events += 1
                continue
            if occurrence_key.startswith(("scheduled:", "morning:")):
                try:
                    delivery_rows = store.list_scheduled_deliveries_for_occurrence(occurrence_key)
                except Exception:
                    invalid_events += 1
                    continue
                delivery_truth = _scheduled_delivery_truth(
                    delivery_rows,
                    job_row=row,
                    job_type=current_job_type,
                    occurrence_key=occurrence_key,
                )
                if delivery_truth == "invalid":
                    invalid_events += 1
                    continue
                if delivery_truth == "failed":
                    status = "failed"
                elif delivery_truth == "ok":
                    status = "ok"
                else:
                    status = "pending"
            if status == "pending":
                if ran_at >= proof_start:
                    pending_events += 1
                continue
            if status == "failed":
                if ran_at >= proof_start:
                    failed_events += 1
                continue
            good_runs.append(ran_at)

        good_runs = sorted(set(good_runs))
        if not good_runs:
            jobs_without_history += 1
            coverage_days.append(0)
            continue

        latest_run = good_runs[-1]
        proof_chain = [run_at for run_at in good_runs if run_at >= proof_start - allowed_gap]
        has_gap = (
            not proof_chain
            or proof_chain[0] > proof_start + allowed_gap
            or now_dt - proof_chain[-1] > allowed_gap
            or any(
                later - earlier > allowed_gap
                for earlier, later in zip(proof_chain, proof_chain[1:])
            )
        )
        covered_start = max(proof_start, proof_chain[0] - allowed_gap) if proof_chain else now_dt
        covered_end = min(now_dt, proof_chain[-1] + allowed_gap) if proof_chain else now_dt
        covered_seconds = max(0.0, (covered_end - covered_start).total_seconds())
        coverage_days.append(min(SCHEDULED_JOB_STREAK_DAYS, int(covered_seconds // 86400)))
        if has_gap:
            cadence_gaps += 1

    if invalid_events:
        return False, f"{invalid_events} invalid run-history event(s) — inspect list scheduled jobs"
    if failed_events:
        return False, f"{failed_events} failed scheduled-job run event(s) recorded — check scheduler daemon/channel health"
    if pending_events:
        return (
            None,
            f"{pending_events} scheduled-job occurrence(s) await terminal delivery proof; wait for outbox recovery",
        )
    if jobs_without_history:
        return (
            None,
            (
                f"{enabled_count}/{SCHEDULED_JOB_EXPECTED_ENABLED_COUNT} expected enabled job(s); "
                f"{jobs_without_history} missing run-history proof; wait for scheduled runs"
            ),
        )
    proven_days = min(coverage_days) if coverage_days else 0
    latest_complete = today
    if cadence_gaps == 0 and proven_days >= SCHEDULED_JOB_STREAK_DAYS:
        return (
            True,
            (
                f"{enabled_count}/{SCHEDULED_JOB_EXPECTED_ENABLED_COUNT} expected enabled job(s); "
                f"{SCHEDULED_JOB_STREAK_DAYS} consecutive complete day(s) proven through {latest_complete.isoformat()}"
            ),
        )
    if cadence_gaps:
        progress = (
            f"retained proof spans up to {proven_days} day(s), but is not consecutive; "
            f"{cadence_gaps} job cadence gap(s)"
        )
    else:
        progress = (
            f"{proven_days} consecutive complete day(s) recorded through "
            f"{latest_complete.isoformat()}; 0 job cadence gap(s)"
        )
    return (
        None,
        (
            f"{enabled_count}/{SCHEDULED_JOB_EXPECTED_ENABLED_COUNT} expected enabled job(s); "
            f"{progress}; need {SCHEDULED_JOB_STREAK_DAYS} consecutive complete day(s)"
        ),
    )


def _check_network_recovery_readiness(config) -> tuple[bool | None, str]:
    """Read only the content-free scheduled-delivery ledger; never claim a live recovery proof."""
    expected_states = {
        "prepared",
        "claimed",
        "sending",
        "accepted",
        "rejected",
        "uncertain",
        "failed",
    }

    def bounded_count(value: object) -> int | None:
        if type(value) is not int or not 0 <= value <= MAX_SCHEDULED_DELIVERY_DIAGNOSTIC_COUNT:
            return None
        return value

    try:
        from jarvis_v2.memory.store import MemoryStore

        snapshot = MemoryStore(config.db_path).scheduled_delivery_operational_snapshot()
    except Exception as exc:
        return False, f"scheduled-delivery recovery ledger unreadable: {_safe_exception(exc)}"
    if not isinstance(snapshot, dict):
        return False, "scheduled-delivery recovery ledger is malformed — inspect list scheduled jobs"

    counts = snapshot.get("state_counts")
    if not isinstance(counts, dict) or set(counts) != expected_states:
        return False, "scheduled-delivery recovery ledger is malformed — inspect list scheduled jobs"
    state_counts = {state: bounded_count(counts.get(state)) for state in expected_states}
    scalar_counts = {
        name: bounded_count(snapshot.get(name))
        for name in (
            "chunk_receipts",
            "occurrences",
            "active_occurrences",
            "uncertain_occurrences",
            "failed_occurrences",
            "orphaned_chunks",
            "content_bearing_chunks",
        )
    }
    counts_saturated = snapshot.get("counts_saturated")
    if (
        any(value is None for value in state_counts.values())
        or any(value is None for value in scalar_counts.values())
        or type(counts_saturated) is not bool
    ):
        return False, "scheduled-delivery recovery ledger is malformed — inspect list scheduled jobs"

    total_chunks = sum(state_counts.values())
    chunk_receipts = scalar_counts["chunk_receipts"]
    occurrences = scalar_counts["occurrences"]
    active_occurrences = scalar_counts["active_occurrences"]
    uncertain_occurrences = scalar_counts["uncertain_occurrences"]
    failed_occurrences = scalar_counts["failed_occurrences"]
    orphaned_chunks = scalar_counts["orphaned_chunks"]
    content_bearing_chunks = scalar_counts["content_bearing_chunks"]
    if (
        counts_saturated
        or chunk_receipts != total_chunks
        or occurrences > chunk_receipts
        or active_occurrences > occurrences
        or uncertain_occurrences > occurrences
        or failed_occurrences > occurrences
        or orphaned_chunks > chunk_receipts
        or content_bearing_chunks > chunk_receipts
    ):
        return False, "scheduled-delivery recovery ledger is inconsistent — inspect list scheduled jobs"

    rejected = state_counts["rejected"]
    if rejected:
        return False, f"{rejected} retryable scheduled-delivery rejection(s) recorded — inspect channel health"
    if uncertain_occurrences:
        return False, f"{uncertain_occurrences} scheduled-delivery outcome(s) need manual review — do not resend automatically"
    if failed_occurrences:
        return False, f"{failed_occurrences} scheduled-delivery failure occurrence(s) recorded — inspect channel health"
    if orphaned_chunks:
        return False, f"{orphaned_chunks} orphaned scheduled-delivery chunk(s) recorded — inspect list scheduled jobs"
    if active_occurrences:
        return None, f"{active_occurrences} scheduled-delivery occurrence(s) still active; wait for terminal recovery state"
    if chunk_receipts == 0:
        return None, "no retained scheduled-delivery recovery evidence — network-loss recovery needs operator-present proof"
    return (
        None,
        (
            f"{chunk_receipts} retained scheduled-delivery receipt(s) have no unresolved recovery state; "
            "network-loss recovery still needs operator-present proof"
        ),
    )


def _command_runs_python_module(command: str, module: str) -> bool:
    try:
        argv = shlex.split(command)
    except ValueError:
        return False
    return argv == ["/opt/homebrew/bin/python3", "-m", module]


def _read_primary_daemon_processes() -> tuple[list[DaemonProcessSnapshot], bool]:
    if sys.platform != "darwin":
        return [], False
    try:
        result = subprocess.run(
            ["/bin/ps", "-axo", "pid=,lstart=,command="],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=3.0,
            check=False,
        )
    except Exception:
        return [], False
    if result.returncode != 0:
        return [], False

    expected_modules = {
        str(contract["module"])
        for contract in DAEMON_SOURCE_FRESHNESS_CONTRACTS.values()
    }
    snapshots: list[DaemonProcessSnapshot] = []
    for line in result.stdout.splitlines():
        parts = line.strip().split(None, 6)
        if len(parts) != 7:
            continue
        command = parts[6]
        module = next(
            (
                candidate
                for candidate in expected_modules
                if _command_runs_python_module(command, candidate)
            ),
            None,
        )
        if module is None:
            continue
        try:
            pid = int(parts[0])
            started_at = datetime.strptime(
                " ".join(parts[1:6]),
                "%a %b %d %H:%M:%S %Y",
            ).timestamp()
            if pid <= 0:
                continue
        except (OverflowError, ValueError):
            continue
        snapshots.append(
            DaemonProcessSnapshot(module=module, started_at_epoch=started_at, pid=pid)
        )
    return snapshots, True


def _read_daemon_source_git_state(
    root: Path,
) -> tuple[DaemonSourceGitSnapshot | None, bool]:
    """Read one bounded, content-free Git HEAD/worktree snapshot."""

    if not root.is_absolute():
        return None, False
    try:
        result = subprocess.run(
            [
                "/usr/bin/git",
                "-C",
                str(root),
                "status",
                "--porcelain=v2",
                "--branch",
                "--untracked-files=normal",
                "--no-ahead-behind",
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=3.0,
            check=False,
        )
    except Exception:
        return None, False
    output = result.stdout
    if (
        result.returncode != 0
        or not isinstance(output, str)
        or len(output) > DAEMON_SOURCE_GIT_STATUS_MAX_CHARS
    ):
        return None, False
    lines = output.splitlines()
    head_lines = [line for line in lines if line.startswith("# branch.oid ")]
    if len(head_lines) != 1:
        return None, False
    head = head_lines[0].removeprefix("# branch.oid ")
    if DAEMON_SOURCE_GIT_HEAD_RE.fullmatch(head) is None:
        return None, False
    clean = all(not line or line.startswith("# ") for line in lines)
    return DaemonSourceGitSnapshot(head=head, clean=clean), True


def _launchctl_block_lines(output: str, name: str) -> tuple[str, ...] | None:
    """Return one flat launchctl-print block without exposing its other fields."""

    lines = output.splitlines()
    starts = [
        index
        for index, line in enumerate(lines)
        if line.strip() == f"{name} = {{"
    ]
    if len(starts) != 1:
        return None
    body: list[str] = []
    depth = 1
    for line in lines[starts[0] + 1 :]:
        stripped = line.strip()
        if stripped.endswith("{"):
            depth += 1
        if stripped == "}":
            depth -= 1
            if depth == 0:
                return tuple(body)
        if depth == 1 and stripped:
            body.append(stripped)
    return None


def _launchctl_loaded_contract(
    output: str,
) -> tuple[
    str,
    tuple[str, ...],
    str,
    str | None,
    str | None,
    str | None,
    tuple[tuple[str, str], ...],
]:
    arguments_block = _launchctl_block_lines(output, "arguments")
    environment_block = _launchctl_block_lines(output, "environment")
    program_matches = re.findall(
        r"(?m)^\s*program\s*=\s*([^\r\n]+?)\s*$",
        output,
    )
    program = program_matches[0] if len(program_matches) == 1 else ""
    arguments = (
        tuple(arguments_block)
        if arguments_block is not None
        and all("=>" not in item and item not in {"{", "}"} for item in arguments_block)
        else ()
    )
    working_matches = re.findall(
        r"(?m)^\s*working directory\s*=\s*([^\r\n]+?)\s*$",
        output,
    )
    working_directory = working_matches[0] if len(working_matches) == 1 else ""
    environment: dict[str, str] = {}
    environment_valid = environment_block is not None
    for item in environment_block or ():
        key, separator, value = item.partition("=>")
        key = key.strip()
        value = value.strip()
        if not separator or not key or key in environment:
            environment_valid = False
            continue
        environment[key] = value
    if not environment_valid:
        environment = {}
    return (
        program,
        arguments,
        working_directory,
        environment.get(V3_DAEMON_ENABLE_ENV),
        environment.get(STARTUP_SELECTED_ENV),
        environment.get(V3_SCHEDULER_ENABLE_ENV),
        tuple(sorted(environment.items())),
    )


def _read_launchd_primary_jobs(
    labels: tuple[str, ...],
) -> tuple[list[LaunchdJobSnapshot], bool]:
    """Inspect launchd job metadata without loading, starting, or changing jobs."""

    if sys.platform != "darwin":
        return [], False
    snapshots: list[LaunchdJobSnapshot] = []
    for label in labels:
        if not isinstance(label, str) or not label:
            return [], False
        try:
            result = subprocess.run(
                ["/bin/launchctl", "print", f"gui/{os.getuid()}/{label}"],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                timeout=3.0,
                check=False,
            )
        except Exception:
            return [], False
        if result.returncode != 0:
            continue
        state_match = re.search(r"(?m)^\s*state\s*=\s*([a-zA-Z-]+)\s*$", result.stdout)
        pid_match = re.search(r"(?m)^\s*pid\s*=\s*([0-9]+)\s*$", result.stdout)
        if state_match is None or pid_match is None:
            return [], False
        try:
            pid = int(pid_match.group(1))
        except ValueError:
            return [], False
        if pid <= 0:
            return [], False
        (
            program,
            program_arguments,
            working_directory,
            daemon_enable,
            selected_environment,
            scheduler_enable,
            environment,
        ) = _launchctl_loaded_contract(result.stdout)
        snapshots.append(
            LaunchdJobSnapshot(
                label=label,
                pid=pid,
                state=state_match.group(1).lower(),
                program=program,
                program_arguments=program_arguments,
                working_directory=working_directory,
                daemon_enable=daemon_enable,
                selected_environment=selected_environment,
                scheduler_enable=scheduler_enable,
                environment=environment,
            )
        )
    return snapshots, True


def _check_daemon_source_freshness(
    root: Path | None = None,
    *,
    process_snapshots: list[DaemonProcessSnapshot] | None = None,
    process_scan_ok: bool | None = None,
    git_reader: Callable[[Path], tuple[DaemonSourceGitSnapshot | None, bool]] | None = None,
) -> tuple[bool | None, str]:
    """Bind daemon start times to a stable clean HEAD and bounded source mtimes.

    This is read-only and content-free: it neither calls launchctl nor returns
    process identifiers, command lines, source paths, or subprocess errors.
    It does not attest the identity of code already loaded into process memory.
    """
    root_path = root or REPO_ROOT
    if process_snapshots is None:
        process_snapshots, scanned = _read_primary_daemon_processes()
        process_scan_ok = scanned
    elif process_scan_ok is None:
        process_scan_ok = True
    if not process_scan_ok:
        return (
            None,
            "daemon process/source freshness unavailable — inspect local process access, then rerun live_check",
        )

    read_git = git_reader or _read_daemon_source_git_state
    try:
        git_before, git_before_ok = read_git(root_path)
    except Exception:
        git_before, git_before_ok = None, False
    if (
        not git_before_ok
        or not isinstance(git_before, DaemonSourceGitSnapshot)
        or type(git_before.head) is not str
        or type(git_before.clean) is not bool
        or DAEMON_SOURCE_GIT_HEAD_RE.fullmatch(git_before.head) is None
    ):
        return (
            False,
            "daemon source Git state unavailable — restore repository access, then rerun live_check",
        )
    if not git_before.clean:
        return (
            False,
            "daemon source worktree is not clean — review local source changes, then rerun live_check",
        )

    by_module: dict[str, list[DaemonProcessSnapshot]] = {}
    for snapshot in process_snapshots:
        if (
            not isinstance(snapshot, DaemonProcessSnapshot)
            or not math.isfinite(snapshot.started_at_epoch)
            or snapshot.started_at_epoch <= 0
        ):
            return False, "daemon process freshness snapshot malformed — rerun live_check"
        by_module.setdefault(snapshot.module, []).append(snapshot)

    missing: list[str] = []
    duplicate: list[str] = []
    stale: list[str] = []
    unreadable: list[str] = []
    for label, contract in DAEMON_SOURCE_FRESHNESS_CONTRACTS.items():
        module = str(contract["module"])
        matching = by_module.get(module, [])
        if not matching:
            missing.append(label)
            continue
        if len(matching) != 1:
            duplicate.append(label)
            continue
        source_mtimes: list[float] = []
        for relative_path in contract["sources"]:
            try:
                source_mtimes.append((root_path / relative_path).stat().st_mtime)
            except Exception:
                unreadable.append(label)
                break
        if label in unreadable:
            continue
        for relative_dir in contract.get("source_dirs", ()):
            source_dir = root_path / relative_dir
            try:
                if not source_dir.is_dir():
                    raise FileNotFoundError
                source_paths = [
                    path
                    for path in source_dir.rglob("*.py")
                    if "__pycache__" not in path.parts and path.is_file()
                ]
                if not source_paths:
                    raise FileNotFoundError
                source_mtimes.extend(path.stat().st_mtime for path in source_paths)
            except Exception:
                unreadable.append(label)
                break
        if source_mtimes and max(source_mtimes) > (
            matching[0].started_at_epoch
            + DAEMON_SOURCE_FRESHNESS_MTIME_GRACE_SECONDS
        ):
            stale.append(label)

    try:
        git_after, git_after_ok = read_git(root_path)
    except Exception:
        git_after, git_after_ok = None, False
    if (
        not git_after_ok
        or not isinstance(git_after, DaemonSourceGitSnapshot)
        or type(git_after.head) is not str
        or type(git_after.clean) is not bool
        or DAEMON_SOURCE_GIT_HEAD_RE.fullmatch(git_after.head) is None
    ):
        return (
            False,
            "daemon source Git state unavailable after inspection — rerun live_check",
        )
    if not git_after.clean or git_after.head != git_before.head:
        return (
            False,
            "daemon source Git state changed during inspection — rerun live_check",
        )

    if unreadable:
        return (
            False,
            f"daemon service source unreadable: {', '.join(sorted(set(unreadable)))} — repair source tree, then rerun live_check",
        )
    if duplicate:
        return (
            False,
            f"multiple primary daemon processes detected: {', '.join(duplicate)} — inspect service state before restart",
        )
    if stale:
        return (
            False,
            f"primary daemon running stale service source: {', '.join(stale)} — restart manually, then rerun live_check",
        )
    if missing:
        return (
            False,
            f"primary daemon not running: {', '.join(missing)} — follow manual recovery, then rerun live_check",
        )
    return (
        True,
        "2/2 primary daemons pass clean stable HEAD and source start-time checks; loaded-memory identity is not attested",
    )


def _valid_public_candidate_root(root: Path) -> bool:
    """Recognize a bound sanitized candidate without exposing marker details."""

    try:
        from jarvis_v2.scripts.public_release_candidate import structural_public_candidate_profile

        return structural_public_candidate_profile(root) is not None
    except Exception:
        return False


def _startup_recovery_documentation_mode(root: Path) -> str | None:
    """Classify only complete private recovery or public deferral wording."""

    try:
        text = (root / "README.md").read_text(encoding="utf-8")
    except Exception:
        return None
    normalized_text = " ".join(text.split()).casefold()

    def contains_exact_semantics(token: str) -> bool:
        return " ".join(token.split()).casefold() in normalized_text

    private_complete = all(
        contains_exact_semantics(token)
        for tokens in STARTUP_PRIVATE_RECOVERY_DOC_CONTRACTS.values()
        for token in tokens
    )
    if private_complete:
        return "private"
    if all(contains_exact_semantics(token) for token in STARTUP_PUBLIC_RECOVERY_BOUNDARY):
        return "sanitized"
    return None


def _private_regular_fd_custody(fd: int) -> bool:
    """Validate private-file custody against one already-open, stable file description."""

    try:
        before = os.fstat(fd)
        after = os.fstat(fd)
    except Exception:
        return False
    return bool(
        stat.S_ISREG(before.st_mode)
        and before.st_uid == os.getuid()
        and before.st_nlink == 1
        and before.st_mode & 0o177 == 0
        and (before.st_dev, before.st_ino, before.st_mode, before.st_uid, before.st_nlink)
        == (after.st_dev, after.st_ino, after.st_mode, after.st_uid, after.st_nlink)
    )


def _same_filesystem_entry(left: os.stat_result, right: os.stat_result) -> bool:
    return left.st_dev == right.st_dev and left.st_ino == right.st_ino


def _open_absolute_directory_fd(path: Path) -> int:
    """Open an absolute directory component-by-component without following links."""

    nofollow = getattr(os, "O_NOFOLLOW", None)
    if nofollow is None or not path.is_absolute():
        raise OSError("stable directory opening is unavailable")
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | nofollow
    )
    current_fd = os.open("/", flags)
    try:
        for component in path.parts[1:]:
            if component in ("", ".", ".."):
                raise OSError("invalid path component")
            next_fd = os.open(component, flags, dir_fd=current_fd)
            os.close(current_fd)
            current_fd = next_fd
        if not stat.S_ISDIR(os.fstat(current_fd).st_mode):
            raise OSError("not a directory")
        return current_fd
    except Exception:
        os.close(current_fd)
        raise


def _launch_agents_directory_custody(fd: int) -> bool:
    try:
        before = os.fstat(fd)
        after = os.fstat(fd)
    except Exception:
        return False
    return bool(
        stat.S_ISDIR(before.st_mode)
        and before.st_uid == os.getuid()
        and before.st_mode & 0o022 == 0
        and _same_filesystem_entry(before, after)
        and before.st_mode == after.st_mode
        and before.st_uid == after.st_uid
    )


def _open_private_regular_fd(path: Path, *, dir_fd: int | None = None) -> int:
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if nofollow is None:
        raise OSError("no-follow file opening is unavailable")
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NONBLOCK", 0)
        | nofollow
    )
    close_parent = False
    if dir_fd is None:
        if not path.is_absolute() or not path.name:
            raise OSError("private path must be absolute")
        parent_fd = _open_absolute_directory_fd(path.parent)
        close_parent = True
        name = path.name
    else:
        if not path.name or str(path) != path.name:
            raise OSError("private entry name is invalid")
        parent_fd = dir_fd
        name = path.name
    try:
        named = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        if stat.S_ISLNK(named.st_mode) or not stat.S_ISREG(named.st_mode):
            raise OSError("private entry is not a regular file")
        fd = os.open(name, flags, dir_fd=parent_fd)
        try:
            opened = os.fstat(fd)
            after = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
            if (
                not _same_filesystem_entry(named, opened)
                or not _same_filesystem_entry(opened, after)
                or not _private_regular_fd_custody(fd)
            ):
                raise OSError("unsafe or unstable private-file custody")
            return fd
        except Exception:
            os.close(fd)
            raise
    finally:
        if close_parent:
            os.close(parent_fd)


def _load_private_plist_at(dir_fd: int, filename: str) -> object:
    """Parse an installed plist from an owner-only, non-linked stable fd."""

    fd = _open_private_regular_fd(Path(filename), dir_fd=dir_fd)
    try:
        before = os.fstat(fd)
        if before.st_size > STARTUP_PLIST_MAX_BYTES:
            raise OSError("private plist exceeds bounded size")
        remaining = STARTUP_PLIST_MAX_BYTES + 1
        chunks: list[bytes] = []
        while remaining:
            chunk = os.read(fd, min(64 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        raw = b"".join(chunks)
        if len(raw) > STARTUP_PLIST_MAX_BYTES:
            raise OSError("private plist exceeds bounded size")
        payload = plistlib.loads(raw)
        after = os.fstat(fd)
        named_after = os.stat(filename, dir_fd=dir_fd, follow_symlinks=False)
        stable_before = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_mode,
            before.st_uid,
            before.st_nlink,
        )
        stable_after = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_mode,
            after.st_uid,
            after.st_nlink,
        )
        if (
            stable_before != stable_after
            or not _same_filesystem_entry(after, named_after)
            or not _private_regular_fd_custody(fd)
        ):
            raise OSError("private-file custody changed during read")
        return payload
    finally:
        os.close(fd)


def _account_home_directory() -> Path:
    """Derive and validate the canonical account home without trusting HOME."""

    account = pwd.getpwuid(os.getuid())
    home = getattr(account, "pw_dir", None)
    if not isinstance(home, str) or not home:
        raise OSError("account home unavailable")
    home_path = Path(home)
    if not home_path.is_absolute() or any(
        component in {"", ".", ".."} for component in home_path.parts[1:]
    ):
        raise OSError("account home malformed")
    info = os.stat(home_path, follow_symlinks=False)
    if (
        not stat.S_ISDIR(info.st_mode)
        or stat.S_ISLNK(info.st_mode)
        or info.st_uid != os.getuid()
    ):
        raise OSError("account home custody invalid")
    return home_path


def _account_launch_agents_directory() -> Path:
    """Derive the installed LaunchAgent root from the account database."""

    return _account_home_directory() / "Library" / "LaunchAgents"


def _account_log_directory() -> Path:
    """Return the canonical future owner-only V3 log directory contract."""

    return _account_home_directory() / STARTUP_LOG_DIRECTORY_RELATIVE


def _owner_only_log_custody(directory: Path, filenames: tuple[str, ...]) -> bool:
    """Check log metadata through no-follow descriptors without reading content."""

    directory_fd: int | None = None
    try:
        directory_fd = _open_absolute_directory_fd(directory)
        directory_info = os.fstat(directory_fd)
        named_directory = os.stat(directory, follow_symlinks=False)
        if (
            not stat.S_ISDIR(directory_info.st_mode)
            or stat.S_ISLNK(named_directory.st_mode)
            or directory_info.st_uid != os.getuid()
            or stat.S_IMODE(directory_info.st_mode) != STARTUP_LOG_DIRECTORY_MODE
            or not _same_filesystem_entry(directory_info, named_directory)
        ):
            return False
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        for filename in filenames:
            if not filename or Path(filename).name != filename:
                return False
            descriptor = os.open(filename, flags, dir_fd=directory_fd)
            try:
                opened = os.fstat(descriptor)
                named = os.stat(filename, dir_fd=directory_fd, follow_symlinks=False)
                if (
                    not stat.S_ISREG(opened.st_mode)
                    or stat.S_ISLNK(named.st_mode)
                    or opened.st_uid != os.getuid()
                    or stat.S_IMODE(opened.st_mode) != STARTUP_LOG_FILE_MODE
                    or opened.st_nlink != 1
                    or not _same_filesystem_entry(opened, named)
                ):
                    return False
            finally:
                os.close(descriptor)
        named_directory_after = os.stat(directory, follow_symlinks=False)
        return bool(
            not stat.S_ISLNK(named_directory_after.st_mode)
            and _same_filesystem_entry(directory_info, named_directory_after)
        )
    except Exception:
        return False
    finally:
        if directory_fd is not None:
            os.close(directory_fd)


def _installed_selected_environment_binding(selector: object) -> str:
    """Validate an installed service's private V3 env selector without rendering it."""

    if not isinstance(selector, str) or not selector.strip():
        return "selector_missing"
    current = os.getenv(STARTUP_SELECTED_ENV, "").strip()
    if not current:
        return "current_unavailable"
    selected = selector.strip()
    if selected != current:
        return "selector_mismatch"
    path = Path(selected)
    if not path.is_absolute():
        return "selector_invalid"
    try:
        fd = _open_private_regular_fd(path)
    except Exception:
        return "selector_invalid"
    try:
        if not _private_regular_fd_custody(fd):
            return "selector_invalid"
    finally:
        os.close(fd)
    return "ready"


def _validate_launchd_primary_observation(
    expected: dict[str, LaunchdJobExpectation],
    jobs: list[LaunchdJobSnapshot],
    launchd_ok: bool,
    processes: list[DaemonProcessSnapshot],
    process_ok: bool,
) -> tuple[bool | None, str, list[DaemonProcessSnapshot]]:
    if not launchd_ok or not process_ok:
        return None, "launchd/process provenance inspection unavailable; activation remains unproven", []

    by_label: dict[str, LaunchdJobSnapshot] = {}
    for snapshot in jobs:
        if (
            not isinstance(snapshot, LaunchdJobSnapshot)
            or snapshot.label not in expected
            or snapshot.label in by_label
            or snapshot.pid <= 0
            or snapshot.state != "running"
        ):
            return None, "activated launchd job state is not proven running; activation remains deferred", []
        by_label[snapshot.label] = snapshot
    if set(by_label) != set(expected):
        return None, "not every activated primary launchd label is loaded and running; activation remains deferred", []

    observed: list[DaemonProcessSnapshot] = []
    for label, module in expected.items():
        job = by_label[label]
        loaded_environment = dict(job.environment)
        expected_environment = dict(module.environment)
        environment_ambiguous = len(loaded_environment) != len(job.environment)
        expected_environment_ambiguous = (
            len(expected_environment) != len(module.environment)
        )
        loaded_extra_keys = set(loaded_environment) - set(expected_environment)
        loaded_extras_invalid = bool(
            loaded_extra_keys - STARTUP_ALLOWED_LOADED_ENVIRONMENT_EXTRAS
        ) or (
            "XPC_SERVICE_NAME" in loaded_extra_keys
            and loaded_environment.get("XPC_SERVICE_NAME") != job.label
        )
        program_binding_invalid = (
            not job.program_arguments
            or job.program != job.program_arguments[0]
        )
        if (
            job.program != module.program
            or program_binding_invalid
            or job.program_arguments != module.program_arguments
            or job.working_directory != module.working_directory
            or job.daemon_enable != module.daemon_enable
            or job.selected_environment != module.selected_environment
            or job.scheduler_enable != module.scheduler_enable
            or environment_ambiguous
            or expected_environment_ambiguous
            or loaded_extras_invalid
            or any(
                loaded_environment.get(key) != value
                for key, value in expected_environment.items()
            )
        ):
            return (
                False,
                "loaded launchd job configuration does not match the installed active contract; reinstall and reload before proof",
                [],
            )
        module_processes = [
            process
            for process in processes
            if isinstance(process, DaemonProcessSnapshot)
            and process.module == module.module
        ]
        if len(module_processes) > 1:
            return (
                False,
                "multiple processes claim an expected primary service module; activation is unsafe",
                [],
            )
        if (
            len(module_processes) != 1
            or module_processes[0].pid != job.pid
            or not math.isfinite(module_processes[0].started_at_epoch)
            or module_processes[0].started_at_epoch <= 0
        ):
            return None, "launchd PID ownership of the expected service process is not proven; activation remains deferred", []
        observed.append(module_processes[0])
    return (
        True,
        f"{len(expected)}/{len(expected)} activated primary launchd jobs own running service processes",
        observed,
    )


def _check_launchd_primary_provenance(
    expected: dict[str, LaunchdJobExpectation],
    *,
    launchd_reader: Callable[[tuple[str, ...]], tuple[list[LaunchdJobSnapshot], bool]] | None = None,
    process_reader: Callable[[], tuple[list[DaemonProcessSnapshot], bool]] | None = None,
) -> tuple[bool | None, str, list[DaemonProcessSnapshot]]:
    """Bind two stable launchd/PID observations to expected service contracts."""

    read_launchd = launchd_reader or _read_launchd_primary_jobs
    read_processes = process_reader or _read_primary_daemon_processes
    try:
        first_jobs, first_launchd_ok = read_launchd(tuple(expected))
        first_processes, first_process_ok = read_processes()
    except Exception:
        return None, "launchd/process provenance inspection unavailable; activation remains unproven", []
    first_ok, first_detail, first_observed = _validate_launchd_primary_observation(
        expected,
        first_jobs,
        first_launchd_ok,
        first_processes,
        first_process_ok,
    )
    if first_ok is not True:
        return first_ok, first_detail, []

    try:
        second_jobs, second_launchd_ok = read_launchd(tuple(expected))
        second_processes, second_process_ok = read_processes()
    except Exception:
        return None, "launchd/process provenance reinspection unavailable; activation remains unproven", []
    second_ok, second_detail, second_observed = _validate_launchd_primary_observation(
        expected,
        second_jobs,
        second_launchd_ok,
        second_processes,
        second_process_ok,
    )
    if second_ok is not True:
        return second_ok, second_detail, []

    first_jobs_canonical = tuple(sorted(first_jobs, key=lambda item: item.label))
    second_jobs_canonical = tuple(sorted(second_jobs, key=lambda item: item.label))
    if first_jobs_canonical != second_jobs_canonical or first_observed != second_observed:
        return (
            None,
            "launchd/process provenance changed during inspection; activation remains unproven",
            [],
        )
    return (
        True,
        f"{len(expected)}/{len(expected)} activated primary launchd jobs own stable running service processes",
        second_observed,
    )


def _check_daemon_startup_readiness(
    root: Path | None = None,
    launch_agents_dir: Path | None = None,
    *,
    launchd_reader: Callable[[tuple[str, ...]], tuple[list[LaunchdJobSnapshot], bool]] | None = None,
    process_reader: Callable[[], tuple[list[DaemonProcessSnapshot], bool]] | None = None,
) -> tuple[bool | None, str]:
    """Read-only: are Jarvis's local startup contracts present and parseable?

    Checked-in LaunchAgent files are deliberately inert templates. An installed
    primary contract counts as activated only when its exact, service-specific
    activation environment is present. Scheduler activation is never inherited
    from Telegram startup readiness. A successful result also uses only the
    read-only ``launchctl print`` verb to bind each activated primary label to
    its observed PID and expected service module. It never controls a process,
    restarts anything, or sends network traffic.
    """
    root_path = root or REPO_ROOT
    missing_wiring: list[str] = []
    drifted_wiring: list[str] = []

    for label, contract in STARTUP_REQUIRED_TEXT.items():
        path = root_path / contract["path"]
        try:
            text = path.read_text(encoding="utf-8")
        except Exception:
            missing_wiring.append(label)
            continue
        if any(token not in text for token in contract["tokens"]):
            drifted_wiring.append(label)

    if missing_wiring or drifted_wiring:
        parts = []
        if missing_wiring:
            parts.append(
                f"missing/unreadable {len(missing_wiring)} launcher contract(s): "
                f"{', '.join(missing_wiring[:3])}"
            )
        if drifted_wiring:
            parts.append(
                f"drifted {len(drifted_wiring)} launcher contract(s): "
                f"{', '.join(drifted_wiring[:3])}"
            )
        return False, "; ".join(parts) + " — repair startup wiring before reboot proof"

    try:
        account_log_directory = _account_log_directory()
    except Exception:
        return (
            False,
            "account home directory is malformed; inspect local log/service custody before reboot proof",
        )

    sanitized_candidate = _valid_public_candidate_root(root_path)
    missing_templates: list[str] = []
    drifted_templates: list[str] = []
    primary_plist_count = 0
    parsed_plist_count = 0
    primary_templates: dict[str, dict[str, object]] = {}
    for label, contract in STARTUP_PLIST_CONTRACTS.items():
        path = root_path / contract["path"]
        try:
            with path.open("rb") as handle:
                plist = plistlib.load(handle)
        except Exception:
            missing_templates.append(label)
            continue
        parsed_plist_count += 1
        if contract.get("primary"):
            primary_plist_count += 1
        if isinstance(plist, dict):
            primary_templates[str(contract["label"])] = plist
        if not isinstance(plist, dict):
            drifted_templates.append(label)
            continue
        program_args = plist.get("ProgramArguments")
        expected_args = ["/opt/homebrew/bin/python3", "-m", contract["module"]]
        program = plist.get("Program")
        expected_log = os.fspath(
            account_log_directory / str(contract["log_filename"])
        )
        expected_template_fields = set(STARTUP_REQUIRED_PLIST_FIELDS)
        if program is not None:
            expected_template_fields.add("Program")
        if "throttle_interval" in contract:
            expected_template_fields.add("ThrottleInterval")
        # A checked-in template must never carry activation or credential keys.
        expected_environment = {
            "PYTHONPATH": str(root_path),
            "PATH": STARTUP_LAUNCHD_PATH,
        }
        if (
            set(plist) != expected_template_fields
            or plist.get("Label") != contract["label"]
            or plist.get("RunAtLoad") is not True
            or plist.get("KeepAlive") is not True
            or program_args != expected_args
            or (program is not None and program != expected_args[0])
            or plist.get("WorkingDirectory") != str(root_path)
            or plist.get("EnvironmentVariables") != expected_environment
            or plist.get("ProcessType") != "Background"
            or plist.get("StandardOutPath") != expected_log
            or plist.get("StandardErrorPath") != expected_log
            or (
                "throttle_interval" in contract
                and plist.get("ThrottleInterval") != contract["throttle_interval"]
            )
        ):
            drifted_templates.append(label)

    documentation_mode = _startup_recovery_documentation_mode(root_path)
    if sanitized_candidate:
        if (
            parsed_plist_count
            or drifted_templates
            or len(missing_templates) != len(STARTUP_PLIST_CONTRACTS)
            or documentation_mode != "sanitized"
        ):
            return (
                False,
                "sanitized candidate startup boundary malformed — rebuild the reviewed candidate",
            )
        return (
            None,
            (
                f"startup/reboot deferred: sanitized candidate has {len(STARTUP_REQUIRED_TEXT)} "
                "validated manual launcher(s) and intentionally omits service templates and private "
                "recovery commands; activation not proven; no process control run"
            ),
        )

    if missing_templates or drifted_templates:
        parts = []
        if missing_templates:
            parts.append(
                f"missing/unreadable {len(missing_templates)} contract(s): "
                f"{', '.join(missing_templates[:3])}"
            )
        if drifted_templates:
            parts.append(
                f"drifted {len(drifted_templates)} contract(s): "
                f"{', '.join(drifted_templates[:3])}"
            )
        return False, "; ".join(parts) + " — repair startup wiring before reboot proof"
    if documentation_mode is None:
        return (
            False,
            "daemon recovery boundary documentation missing or malformed — repair the bounded recovery contract before reboot proof",
        )

    primary_labels = [
        str(contract["label"])
        for contract in STARTUP_PLIST_CONTRACTS.values()
        if contract.get("primary")
    ]
    primary_installed = 0
    primary_activated = 0
    installed_unreadable = 0
    installed_drift_fields: set[str] = set()
    activation_invalid = 0
    environment_not_allowlisted = 0
    scheduler_disable_missing = 0
    scheduler_activation_invalid = 0
    activation_incomplete = 0
    selector_missing = 0
    selector_mismatched = 0
    selector_invalid = 0
    selector_current_unavailable = 0
    activated_contracts: dict[str, LaunchdJobExpectation] = {}
    install_state = "install status unknown"
    try:
        agents_dir = launch_agents_dir or _account_launch_agents_directory()
        if not agents_dir.is_absolute():
            return (
                False,
                "installed LaunchAgent directory is malformed; inspect local service custody before reboot proof",
            )
        try:
            agents_dir_fd = _open_absolute_directory_fd(agents_dir)
        except FileNotFoundError:
            agents_dir_fd = None
        except Exception:
            return (
                False,
                "installed LaunchAgent directory is malformed; inspect local service custody before reboot proof",
            )
        if agents_dir_fd is not None and not _launch_agents_directory_custody(agents_dir_fd):
            os.close(agents_dir_fd)
            return (
                False,
                "installed LaunchAgent directory has unsafe custody; repair local service custody before reboot proof",
            )
        try:
            for contract in STARTUP_PLIST_CONTRACTS.values():
                is_primary = bool(contract.get("primary"))
                if agents_dir_fd is None:
                    continue
                try:
                    installed_plist = _load_private_plist_at(
                        agents_dir_fd,
                        contract["path"].name,
                    )
                except FileNotFoundError:
                    continue
                except Exception:
                    installed_unreadable += 1
                    continue
                if is_primary:
                    primary_installed += 1
                template = primary_templates.get(str(contract["label"]), {})
                if not isinstance(installed_plist, dict) or not isinstance(template, dict):
                    installed_unreadable += 1
                    continue

                contract_drifted = False
                if set(installed_plist) != set(template):
                    installed_drift_fields.add("TopLevelKeys")
                    contract_drifted = True
                for field in STARTUP_CRITICAL_PLIST_FIELDS:
                    if field == "EnvironmentVariables":
                        continue
                    if installed_plist.get(field) != template.get(field):
                        installed_drift_fields.add(field)
                        contract_drifted = True

                installed_environment = installed_plist.get("EnvironmentVariables")
                template_environment = template.get("EnvironmentVariables")
                required_activation = contract.get("activation_environment", {})
                if (
                    not isinstance(installed_environment, dict)
                    or not isinstance(template_environment, dict)
                    or not isinstance(required_activation, dict)
                ):
                    installed_drift_fields.add("EnvironmentVariables")
                    continue
                allowed_keys = (
                    set(template_environment)
                    | set(required_activation)
                    | {STARTUP_SELECTED_ENV}
                )
                unexpected_keys = set(installed_environment) - allowed_keys
                if unexpected_keys:
                    environment_not_allowlisted += 1
                    contract_drifted = True
                if any(
                    installed_environment.get(key) != value
                    for key, value in template_environment.items()
                ):
                    installed_drift_fields.add("EnvironmentVariables")
                    contract_drifted = True

                activation_keys = set(required_activation) | {STARTUP_SELECTED_ENV}
                activation_requested = bool(set(installed_environment) & activation_keys)
                if activation_requested and V3_SCHEDULER_ENABLE_ENV not in installed_environment:
                    scheduler_disable_missing += 1
                    contract_drifted = True
                if (
                    V3_SCHEDULER_ENABLE_ENV in installed_environment
                    and installed_environment.get(V3_SCHEDULER_ENABLE_ENV) != "0"
                ):
                    scheduler_activation_invalid += 1
                    contract_drifted = True
                if activation_requested and not set(required_activation).issubset(installed_environment):
                    activation_incomplete += 1
                    contract_drifted = True

                activation_missing = False
                for key, expected_value in required_activation.items():
                    if key not in installed_environment:
                        activation_missing = True
                    elif installed_environment.get(key) != expected_value:
                        activation_invalid += 1
                        contract_drifted = True
                activation_exact = not contract_drifted and not activation_missing
                if activation_exact:
                    selector_state = _installed_selected_environment_binding(
                        installed_environment.get(STARTUP_SELECTED_ENV)
                    )
                    if selector_state == "selector_missing":
                        selector_missing += 1
                    elif selector_state == "selector_mismatch":
                        selector_mismatched += 1
                    elif selector_state == "selector_invalid":
                        selector_invalid += 1
                    elif selector_state == "current_unavailable":
                        selector_current_unavailable += 1
                    else:
                        if is_primary:
                            primary_activated += 1
                        activated_contracts[str(contract["label"])] = LaunchdJobExpectation(
                            module=str(contract["module"]),
                            program=str(
                                installed_plist.get("Program")
                                or installed_plist["ProgramArguments"][0]
                            ),
                            program_arguments=tuple(installed_plist["ProgramArguments"]),
                            working_directory=str(installed_plist["WorkingDirectory"]),
                            daemon_enable=str(installed_environment[V3_DAEMON_ENABLE_ENV]),
                            selected_environment=str(installed_environment[STARTUP_SELECTED_ENV]),
                            scheduler_enable=str(installed_environment[V3_SCHEDULER_ENABLE_ENV]),
                            environment=tuple(sorted(installed_environment.items())),
                        )
        finally:
            if agents_dir_fd is not None:
                os.close(agents_dir_fd)
        install_state = f"{primary_installed}/{len(primary_labels)} primary LaunchAgent installed"
    except Exception:
        return (
            False,
            "installed LaunchAgent inspection unavailable; inspect local service custody before reboot proof",
        )

    if installed_unreadable:
        return (
            False,
            "installed LaunchAgent unreadable/malformed; reinstall validated contract before reboot proof",
        )
    if scheduler_activation_invalid:
        return (
            False,
            "installed LaunchAgent scheduler-disable value is invalid; exact value 0 is required",
        )
    if scheduler_disable_missing:
        return (
            False,
            "activated LaunchAgent is missing the explicit scheduler-disable value; exact value 0 is required",
        )
    if activation_incomplete:
        return (
            False,
            "installed LaunchAgent activation environment is incomplete; reinstall the validated active contract",
        )
    if environment_not_allowlisted:
        return (
            False,
            "installed LaunchAgent environment is not allowlisted; reinstall validated contract before reboot proof",
        )
    if activation_invalid:
        return (
            False,
            "installed LaunchAgent activation value is invalid; exact supervised activation is required",
        )
    if selector_missing:
        return (
            False,
            "activated LaunchAgent is missing the selected V3 environment binding; reinstall the validated contract",
        )
    if selector_mismatched:
        return (
            False,
            "activated LaunchAgent selected V3 environment does not match this diagnostic session",
        )
    if selector_invalid:
        return (
            False,
            "activated LaunchAgent selected V3 environment has unsafe custody; repair it before reboot proof",
        )
    if selector_current_unavailable:
        return (
            None,
            "activated LaunchAgent environment binding is not proven because this diagnostic has no selected V3 environment",
        )
    if installed_drift_fields:
        sorted_fields = sorted(installed_drift_fields)
        field_list = ", ".join(sorted_fields[:3])
        if len(sorted_fields) > 3:
            field_list += f" (+{len(sorted_fields) - 3})"
        return (
            False,
            f"installed LaunchAgent drifted field(s): {field_list}; reinstall before reboot proof",
        )

    if activated_contracts and root is None and launch_agents_dir is None:
        activated_log_filenames = tuple(
            str(contract["log_filename"])
            for contract in STARTUP_PLIST_CONTRACTS.values()
            if str(contract["label"]) in activated_contracts
        )
        if not _owner_only_log_custody(
            account_log_directory,
            activated_log_filenames,
        ):
            return (
                False,
                "activated LaunchAgent log targets have unsafe custody; repair owner-only logs before reboot proof",
            )

    doc_state = (
        "private recovery contract present"
        if documentation_mode == "private"
        else "public restart boundary present"
    )
    detail = (
        f"startup wiring ready: {len(STARTUP_REQUIRED_TEXT)} launcher(s), "
        f"{parsed_plist_count} inert LaunchAgent template(s), {primary_plist_count} primary; "
        "RunAtLoad+KeepAlive contracts parse; Telegram daemon owns scheduler ticker but scheduler "
        "activation remains separate; "
        f"{install_state}; {primary_activated}/{len(primary_labels)} explicitly activated; "
        f"{doc_state}; no process control run"
    )
    if primary_activated == len(primary_labels):
        provenance_ok, provenance_detail, observed_processes = _check_launchd_primary_provenance(
            activated_contracts,
            launchd_reader=launchd_reader,
            process_reader=process_reader,
        )
        if provenance_ok is not True:
            return provenance_ok, provenance_detail
        detail += f"; {provenance_detail}"
        if root is None and launch_agents_dir is None:
            freshness_ok, freshness_detail = _check_daemon_source_freshness(
                root_path,
                process_snapshots=observed_processes,
                process_scan_ok=True,
            )
            if freshness_ok is not True:
                return freshness_ok, freshness_detail
            detail += f"; {freshness_detail}"
        return True, detail + " — live reboot proof still needs operator present"
    return None, detail + " — service activation/reboot deferred and not proven"


def _check_telegram(send: bool, *, probe: bool = False) -> tuple[bool | None, str]:
    try:
        # Match main() so direct, bounded owner-channel checks see the project .env.
        load_config()
        from jarvis_v2.automations import telegram_control as tg

        owner = tg._owner_chat_id()
        token = tg._bot_token()
    except Exception as e:
        return False, f"EXCEPTION: {_safe_exception(e)}"
    if not token:
        return False, "no bot token — set JARVIS_TELEGRAM_BOT_TOKEN"
    if not owner:
        return False, "no owner chat id — set JARVIS_OWNER_TELEGRAM"
    if not send and not probe:
        return True, (
            "token + owner chat id present "
            "(set JARVIS_LIVE_CHECK_TELEGRAM_READ=1 for read-only API check)"
        )
    if probe and not send:
        try:
            res = tg._api_call(token, "getMe", {}, 10.0)
        except Exception as e:
            return False, (
                f"read-only API check unavailable ({_safe_exception(e)}); "
                "run `setup check`, verify network/TLS setup, then retry"
            )
        if not isinstance(res, dict) or res.get("ok") is not True:
            error = _safe_detail(res.get("error", "unknown")) if isinstance(res, dict) else "invalid response"
            return False, (
                f"read-only API check failed ({error}); run `setup check`, "
                "verify the BotFather token and network/TLS setup, then retry"
            )
        identity = res.get("result")
        if not isinstance(identity, dict) or not identity.get("id"):
            return False, (
                "read-only API check returned no bot identity; verify the BotFather token, "
                "run `setup check`, then retry"
            )
        return True, "read-only Telegram API identity check succeeded; no message sent"
    try:
        res = tg.send_message(owner, "✅ Jarvis live_check: Telegram owner channel is working.")
    except Exception as e:
        return False, f"send EXCEPTION: {_safe_exception(e)}"
    if not res.get("ok"):
        return False, f"send failed: {_safe_detail(res.get('error', 'unknown'))}"
    return True, "Telegram API accepted test message for owner chat id"


def main() -> None:
    if os.getenv("JARVIS_LIVE_CHECK") != "1":
        print(
            "Live check is opt-in. Run with:\n"
            f"  {V3_LIVE_CHECK_COMMAND}"
        )
        return

    argv = sys.argv[1:]
    contacts = _arg_contacts(argv)
    send = os.getenv("JARVIS_LIVE_CHECK_SEND") == "1"
    telegram_read_probe = os.getenv("JARVIS_LIVE_CHECK_TELEGRAM_READ") == "1"
    config, config_error = _load_config_for_check()
    config_unavailable = (False, config_error or "config unavailable")

    daily_brief = _check_daily_brief(config) if config is not None else config_unavailable
    research = _check_research_readiness(config) if config is not None else config_unavailable
    personal_routes = _check_personal_route_readiness()
    chat_path = _check_chat_path_health(config) if config is not None else config_unavailable
    mixed_conversation = _check_mixed_conversation_health(config) if config is not None else config_unavailable
    personal_integrations = _check_personal_integration_readiness(config) if config is not None else config_unavailable
    personal_proofs = _check_personal_proof_history(config) if config is not None else config_unavailable
    morning_brief = _check_morning_brief_schedule(config) if config is not None else config_unavailable
    job_freshness = _check_scheduled_job_freshness(config) if config is not None else config_unavailable
    job_streak = _check_scheduled_job_streak(config) if config is not None else config_unavailable
    channel_health = _check_channel_health(config) if config is not None else config_unavailable
    call_readiness = (
        _check_call_readiness(config, contacts)
        if config is not None
        else config_unavailable
    )
    daemon_startup = _check_daemon_startup_readiness()
    network_recovery = _check_network_recovery_readiness(config) if config is not None else config_unavailable

    print(f"{'FEATURE':<20} {'STATUS':<8} DETAIL")
    print("-" * 78)
    passed = 0
    testable = 0
    for label, (ok, detail) in (
        ("Jarvis config", (config is not None, "loaded" if config is not None else config_error or "config unavailable")),
        ("macOS Contacts", _check_contact_matrix(contacts, _check_contacts, label="contact lookup(s)")),
        ("send resolution", _check_contact_matrix(contacts, _check_send_resolution, label="send preflight(s)")),
        ("call readiness", call_readiness),
        ("voice transcription", _check_voice_transcription()),
        ("voice warmup", _check_voice_warmup()),
        ("dashboard voice", _check_dashboard_voice_wiring()),
        ("local talk", _check_local_talk_readiness()),
        ("telegram voice", _check_telegram_voice_readiness()),
        ("image OCR", _check_image_ocr()),
        ("research", research),
        ("daily brief", daily_brief),
        ("chat path", chat_path),
        ("mixed conversation", mixed_conversation),
        ("personal integrations", personal_integrations),
        ("personal proofs", personal_proofs),
        ("personal routes", personal_routes),
        ("morning brief job", morning_brief),
        ("jobs freshness", job_freshness),
        ("jobs 7-day proof", job_streak),
        ("daemon startup", daemon_startup),
        ("network recovery", network_recovery),
        ("acceptance coverage", _check_acceptance_harness_coverage()),
        ("acceptance gaps", _check_acceptance_open_items()),
        ("acceptance next", _check_acceptance_next_action()),
        ("aggregate smoke", _check_aggregate_smoke_proof()),
        ("operator evals", _check_operator_workflow_eval_proof()),
        ("error guidance", _check_error_guidance_readiness()),
        ("channel health", channel_health),
        ("guardrail plane", _check_guardrail_control_plane()),
        ("proof ledger", _check_proof_ledger_control_plane()),
        ("completion gate", _check_completion_phone_gate()),
        ("learning loop", _check_learning_recovery_control_plane()),
        ("phone approvals", _check_phone_approval_flow()),
        ("phone control", _check_phone_control_center()),
        ("Telegram owner", _check_telegram(send, probe=telegram_read_probe)),
    ):
        if ok is not None:
            testable += 1
        passed += _row(label, ok, detail)
    print("-" * 78)
    print(f"{passed}/{testable} live feature(s) working" + ("" if contacts else "  (Contacts skipped)"))
    # Diagnostic: always exit 0 so it never breaks automation.
    sys.exit(0)


if __name__ == "__main__":
    main()
