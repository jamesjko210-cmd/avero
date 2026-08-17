from __future__ import annotations

from collections import Counter, defaultdict
import re
from typing import Any, Callable

from jarvis_v2.agent.types import ToolResult


TOOLSET_LABELS = {
    "approvals": "approval queue",
    "audit": "audit trail",
    "brain": "GBrain-style memory",
    "browser": "browser research",
    "code": "code and shell",
    "computer": "computer control",
    "continuity": "continuity",
    "conversation": "conversation memory",
    "decisions": "decisions",
    "files": "files",
    "feedback": "feedback learning",
    "goals": "goals and projects",
    "ingest": "Obsidian ingest",
    "learning": "learning review",
    "memory": "memory curation",
    "notes": "Obsidian notes",
    "organize": "brain-dump organizing",
    "people": "people memory",
    "personal": "personal integrations",
    "preferences": "preferences",
    "proactive": "proactive briefs",
    "profile": "profile",
    "safety": "safety and autonomy",
    "scheduler": "scheduled jobs",
    "skills": "skills",
    "state": "state snapshots",
    "system": "system",
    "utilities": "utilities",
    "voice": "voice",
}
LOCAL_PATH_RE = re.compile(r"/(?:Users|private|var/folders|tmp)/[^\n\r;]*")

STARTER_COMMANDS = {
    "core": ["priority goal", "harness status", "harness doctrine", "coding discipline: improve Jarvis recovery tests", "harness completion", "completion audit", "evidence ledger", "completion claim gate", "completion next proof", "completion proof refresh", "harness readiness digest", "execution proof bundle: organize downloads; verification list changed files; tests smoke passed; evidence recent run ok; recovery stop on mismatch", "execution mission control: organize downloads and summarize what changed", "execution case handoff: organize downloads and summarize what changed", "save execution case: organize downloads and summarize what changed", "case evidence packet latest: verification receipt 12 confirmed changed files", "case evidence latest: verification receipt 12 confirmed changed files", "execution case latest", "execution case gate", "execution case review", "execution case closure", "execution case timeline", "execution runbook: organize downloads and summarize what changed", "agi gates", "agi next build move", "agi next build move: personal integrations", "harness cycle: organize my desktop and summarize what changed", "harness lifecycle: inspect my screen and summarize what changed", "harness control: organize downloads and summarize what changed", "command cockpit: organize downloads and summarize what changed", "harness operations", "architecture map", "roadmap", "brain loop", "model routing status", "specialist router contract: summarize this work and propose the next code test", "specialist route quality: summarize this work and propose the next code test", "specialist execution readiness: fix this code bug and run focused tests", "specialist handoff receipt: fix this code bug and run focused tests", "specialist model draft: summarize this work history into a brief", "specialist cycle ledger: summarize memory; tool brain_search; args query=Jarvis harness; verification cites local memory only; runtime_trace reviewed; runtime_trace_sha256 <hash>; verification_receipt reviewed; verification_receipt_sha256 <hash>; audit reviewed; audit_sha256 <hash>; recovery reviewed; recovery_sha256 <hash>; learning reviewed; learning_sha256 <hash>; completion_claim reviewed; completion_claim_sha256 <hash>", "model planner prompt preview: find files README in .", "capability map", "status dashboard", "verification receipt", "runtime trace receipt", "execution audit gate", "execution health report", "recovery closure checklist", "execution learning closure", "after-action learning packet", "acceptance gate: organize downloads; evidence recent tool run ok; tests smoke passed", "jarvis status"],
    "safety": ["prototype readiness", "setup check", "storage status", "storage recovery plan", "storage recovery check", "voice setup check", "computer control status", "safety status", "privacy report", "readiness report", "second loop packet: clean up my desktop", "rehearse: run command python3 --version", "risky request lifecycle: run command python3 --version", "action readiness: run a script and email me the result", "execution contract: organize my downloads and summarize what changed", "argument contract: run command python3 --version", "verification packet: organize my downloads and summarize what changed", "acceptance gate: organize downloads; evidence recent tool run ok; tests smoke passed", "command intake: organize downloads and summarize what changed", "command cockpit: organize downloads and summarize what changed", "execution mission control: organize downloads and summarize what changed", "execution case handoff: organize downloads and summarize what changed", "save execution case: organize downloads and summarize what changed", "case evidence packet latest: approval readiness, approval packet, and approval chain proof reviewed", "case evidence latest: approval readiness, approval packet, and approval chain proof reviewed; verification receipt linked", "execution case latest", "execution case gate", "execution case review", "execution case closure", "execution case timeline", "execution runbook: organize downloads and summarize what changed", "execution readiness matrix: organize my downloads and summarize what changed", "dispatch decision: organize my downloads and summarize what changed", "planner gap: organize my downloads and summarize what changed", "autonomy plan: clean up my desktop"],
    "approvals": ["pending approvals", "approval review", "approval history", "approval chain proof 1", "approval readiness 1", "approval detail 1", "approval packet 1", "save approval review", "approve approval 1", "dismiss approval 1"],
    "continuity": ["return brief", "handoff brief", "session closeout", "save session closeout", "work queue", "save work queue", "build progress", "build delta", "work block checkpoint", "operator instruction supersession: previous=<older instruction> latest=<newer instruction> stop_at=<ISO> current_time=<ISO>", "checkpoint recovery", "checkpoint recovery apply", "checkpoint recovery receipt", "checkpoint recovery execute reviewed=true step=<reviewed local-safe step> verification=<evidence>", "checkpoint recovery follow-through: step=<reviewed local-safe step> verification=<evidence> receipt=<receipt path> receipt_sha256=<hash> checkpoint=<checkpoint path> checkpoint_sha256=<hash> stop=<stop condition>", "autonomy resume gate: stop_at=<ISO> current_time=<ISO> step=<reviewed local-safe step> verification=<evidence> receipt=<path> receipt_sha256=<hash> checkpoint=<path> checkpoint_sha256=<hash> stop_condition=<condition>", "autonomy continuation execution: next_step=<one local-safe step> next_verification=<evidence> receipt_sha256=<hash> checkpoint_sha256=<hash> prior_cycle_ledger_token_sha256=<optional prior proof hash>", "autonomy step closure: step=<completed local-safe step> post_step_verification=<evidence> post_step_receipt_path=<path> post_step_receipt_sha256=<hash> post_step_checkpoint_path=<path> post_step_checkpoint_sha256=<hash> prior_cycle_ledger_token_sha256=<optional prior proof hash>", "autonomy cycle ledger: step=<completed local-safe step> verification=<evidence> receipt=<path> receipt_sha256=<hash> checkpoint=<path> checkpoint_sha256=<hash> next_step=<one local-safe step> post_step_verification=<evidence> post_step_receipt_sha256=<hash> post_step_checkpoint_sha256=<hash> prior_cycle_ledger_token_sha256=<optional prior proof hash>", "save build progress", "save build delta", "save work block checkpoint", "recent saved notes", "focus brief", "work session packet", "safe next actions", "next action packet", "what should I do now", "send me a priority card", "priority stack", "continuation packet", "build target packet", "harness build slice: personal connector readiness", "harness build slice: dashboard interface polish", "save build target packet", "mission control"],
    "memory": ["remember that ...", "search memory for ...", "memory stats", "personal context status", "learning review", "knowledge promotion packet <id>", "session learning preview", "save learning review", "queue learning tasks", "feedback: Jarvis should be more concise", "failure to test: Jarvis overlapped dashboard text", "failure clusters", "failure promotion packet", "feedback report", "save feedback report", "feedback actions", "save feedback actions"],
    "notes": ["list jarvis notes", "show my notes", "list jarvis notes in Projects", "search jarvis notes for project plan", "outline jarvis note Projects/Idea", "read jarvis note Projects/Note Name", "create jarvis note Projects/Idea: ...", "append jarvis note Projects/Idea: ...", "save jarvis note Projects/Idea: ...", "save to jarvis notes: ...", "write this down: ...", "write down: ...", "take a note: ...", "make note of ...", "note to self: ..."],
    "brain": ["brain search: Jarvis safety", "brain think: what should Jarvis remember about safety?", "brain graph", "brain neighbors 1"],
    "learning": ["learning review", "session learning preview", "after-action learning packet", "execution learning closure", "execution health report", "recovery closure checklist", "failure to test: Jarvis overlapped dashboard text", "failure clusters", "failure promotion packet", "failure patch receipt: layout; changed files jarvis_v2/scripts/smoke_test_status_server.py; test first pre-patch failing assertion reviewed; verification python3 -m jarvis_v2.scripts.smoke_test_status_server passed; compile python3 -m compileall -q jarvis_v2 passed; rollback scoped test assertion; patch receipt sha256 <hash>", "failure patch application bridge: layout; changed files jarvis_v2/scripts/smoke_test_status_server.py; test first pre-patch failing assertion reviewed; verification python3 -m jarvis_v2.scripts.smoke_test_status_server passed; compile python3 -m compileall -q jarvis_v2 passed; rollback scoped test assertion; patch receipt sha256 <hash>; contract apply contract reviewed; application bridge applied patch bound to contract", "failure learning record: layout; changed files jarvis_v2/scripts/smoke_test_status_server.py; test first pre-patch failing assertion reviewed; verification python3 -m jarvis_v2.scripts.smoke_test_status_server passed; compile python3 -m compileall -q jarvis_v2 passed; rollback scoped test assertion; patch receipt sha256 <hash>; contract apply contract reviewed; application bridge applied patch bound to contract; completion audit reviewed; evidence ledger reviewed; completion claim gate reviewed; review post-claim review complete; after-action learning reviewed; record target learning review; regression smoke_test_status_server_layout linked; durability checkpoint saved; learning record sha256 <hash>", "failure learning closure ledger: layout; changed files jarvis_v2/scripts/smoke_test_status_server.py; test first pre-patch failing assertion reviewed; verification python3 -m jarvis_v2.scripts.smoke_test_status_server passed; compile python3 -m compileall -q jarvis_v2 passed; rollback scoped test assertion; patch receipt sha256 <hash>; contract apply contract reviewed; application bridge applied patch bound to contract; completion audit reviewed; evidence ledger reviewed; completion claim gate reviewed; review post-claim review complete; after-action learning reviewed; record target learning review; regression smoke_test_status_server_layout linked; durability checkpoint saved; learning record sha256 <hash>", "save learning review", "queue learning tasks", "feedback report", "feedback actions", "weak memories", "duplicate memories"],
    "feedback": ["feedback: Jarvis should be more concise", "failure to test: Jarvis overlapped dashboard text", "failure clusters", "failure promotion packet", "feedback report", "save feedback report", "feedback actions", "save feedback actions"],
    "skills": ["skill match preview: import notes into memory", "extract linked skills from https://github.com/NousResearch/hermes-agent https://github.com/tinyhumansai/openhuman", "install linked skills from hermes-agent openhuman", "save skill ... when ... do ...", "search skills memory", "list skills", "get skill ..."],
    "conversation": ["assistant turn rehearsal: can we talk about memory?", "chat continuity brief", "chat response health", "chat context: what should Jarvis do next?", "chat prompt preview: how should Jarvis talk about memory?", "save chat context: how should Jarvis use memory?", "recent conversation", "session learning preview", "summarize this session", "draft skill from this session called ..."],
    "computer": ["computer control status", "computer readiness: open settings", "computer task plan: open settings", "computer action packet: click x 100 y 200 expectation settings opens", "approved screen observation receipt: expectation settings opens; observation settings window is visible; source approved_screenshot; observation id obs-7; screenshot path /tmp/jarvis_oav_before.png; captured at 2026-06-09T10:00:00+09:00; redaction reviewed safe; approval 7", "screen observation freshness: expectation settings opens; observation settings window is visible; source approved_screenshot; observation id obs-7; screenshot path /tmp/jarvis_oav_before.png; captured at 2026-06-09T10:00:00+09:00; current time 2026-06-09T10:03:00+09:00; max age 300; redaction reviewed safe; approval 7", "screen observation confidence: expectation settings opens; observation settings window is visible; source approved_screenshot", "screen vision prompt preview: expectation settings opens; observation settings window is visible; source approved_screenshot; observation id obs-7; screenshot path /tmp/jarvis_oav_before.png; captured at 2026-06-09T10:00:00+09:00; redaction reviewed safe; approval 7", "screen vision model review: expectation settings opens; observation settings window is visible; source approved_screenshot; observation id obs-7; screenshot path /tmp/jarvis_oav_before.png; captured at 2026-06-09T10:00:00+09:00; redaction reviewed safe; approval 7; consent=true", "screen verification contract: expectation settings opens; observation settings window is visible", "observe act verify proof: action click; x 100; y 200; expectation settings opens; observation settings opens; source approved_screenshot", "oav route lock: action click; x 100; y 200; expectation settings opens; observation settings opens; source approved_screenshot; after settings opens; approval readiness reviewed; approval packet reviewed; approval chain proof passed; approved rerun verified", "oav approval bridge: action click; x 100; y 200; expectation settings opens; observation settings opens; source approved_screenshot; after settings opens; approval readiness reviewed; approval packet reviewed; approval chain proof passed; approved rerun verified; approval 7; approved run 22; verification receipt 22", "oav cycle ledger: action click; x 100; y 200; expectation settings opens; observation settings opens; source approved_screenshot; after settings opens; approval 7; approved run 22; verification receipt 22; audit execution audit passed; health execution health passed; review operator reviewed final packet; verification verification receipt 22 verified; post health execution health passed; post audit execution audit passed; learning after-action learning reviewed", "enable computer control", "observe screen"],
    "goals": ["new goal get fit because health", "show my goals", "what goals do I have", "goal 1 status", "next actions", "what are my next actions", "add step to goal 1: ...", "complete goal step 2"],
    "tasks": ["tasks", "show my tasks", "todo list", "what do I need to do", "add todo buy milk", "tasks due today", "what's due tomorrow", "today's tasks", "completed tasks", "overdue tasks", "show high priority tasks", "urgent tasks", "task overview", "task board", "todo board", "next task", "show next task", "what is my next task", "task completion packet 1", "complete task 1 with evidence: verified in recent tool runs", "show task 1", "search tasks approval", "export tasks"],
    "code": ["run command python3 --version", "read README.md (approval-gated)", "read file README.md (approval-gated)", "find my receipt file (approval-gated)", "find files README in . (approval-gated)"],
    "personal": ["integration status", "integration readiness", "integration execution matrix", "integration execution matrix: email", "integration adapter manifest", "integration adapter manifest: email", "integration adapter probe: email -> search mailbox metadata; target inbox; time last 7 days; data metadata-only; verification fake rows only", "integration adapter acceptance: email", "integration contract: email", "integration action preview: email -> send draft reply", "integration scope packet: email -> search mailbox metadata; target inbox; time last 7 days; data metadata-only", "integration dry run contract: email -> search mailbox metadata; target inbox; time last 7 days; data metadata-only; verification fake rows only", "integration runbook: email -> send draft reply; target thread; time today; data draft-only; verification confirm not sent; rollback discard draft", "integration promotion gate: email -> send draft reply; target thread; time today; data draft-only; verification confirm not sent; rollback discard draft; tests blocked send smoke; audit tool run receipt", "integration implementation spec: email -> send draft reply; target thread; time today; data draft-only; verification confirm not sent; rollback discard draft; tests blocked send smoke; audit tool run receipt", "integration preflight contract: email -> send draft reply; target thread; time today; data draft-only; verification confirm not sent; rollback discard draft; tests blocked send smoke; audit tool run receipt", "integration proof bundle: email -> search mailbox metadata; target inbox; time last 7 days; data metadata-only; verification fake rows only; rollback no connector call; tests adapter acceptance passed; audit adapter acceptance receipt; acceptance gate passed", "integration implementation review: email -> search mailbox metadata; target inbox; time last 7 days; data metadata-only; verification fake rows only; rollback no connector call; tests adapter acceptance passed; audit adapter acceptance receipt; acceptance gate passed; status status api smoke passed", "integration route lock: email -> search mailbox metadata; target inbox; time last 7 days; data metadata-only; verification fake rows only; rollback no connector call; tests adapter acceptance passed; audit adapter acceptance receipt; acceptance gate passed; status status api smoke passed", "legacy connector migration audit", "integration migration plan: calendar", "what's on my calendar today", "what is on my schedule today", "what am I doing today", "schedule today", "free today", "schedule PTO all day tomorrow", "move event evt123 to June 20 at 2pm", "calendar edits need an exact event id", "translate hello to Korean", "how do you say thank you in Korean", "convert 100 USD to KRW", "weather in Tokyo", "news about Korea", "what is bitcoin worth", "stock price of AAPL", "remind me to review Jarvis"],
    "scheduler": [
        "scheduler context refresh",
        "schedule assistant basics",
        "resume job State Snapshot",
        "resume job Conversation Compaction",
        "list scheduled jobs",
        "run due jobs",
    ],
    "proactive": ["morning startup", "save morning startup", "daily plan", "daily brief", "weekly review context", "weekly review prompt preview", "weekly review"],
    "voice": ["voice setup check", "voice capture privacy", "voice reply preview: Jarvis is ready", "spoken turn rehearsal: run command python3 --version", "voice file transcription plan: /path/to/audio.m4a", "voice audio file gate: /path/to/audio.m4a consent=true", "voice audio file transcribe: /path/to/audio.m4a consent=true receipt_id=voice-file-...", "photo intake plan: /path/to/receipt.png", "review this photo", "what does this receipt say", "ocr image: /path/to/receipt.png", "voice input plan push-to-talk", "voice transcript review: run command python3 --version", "voice confirmation: remember that Jarvis should ask before acting on speech", "voice confirmation receipt: run command python3 --version confirmed=true", "voice confirmation audit ledger: run command python3 --version; confirmed=true; privacy_receipt_id=<receipt>; receipt_id=<receipt>; receipt_nonce=<nonce>", "voice route gate: run command python3 --version confirmed=true", "voice command lifecycle: run command python3 --version", "voice cycle ledger: run command python3 --version; confirmed=true; verification reviewed; verification_receipt_sha256=<hash>; post health reviewed; execution_health_sha256=<hash>; post audit reviewed; execution_audit_sha256=<hash>; learning reviewed; after_action_learning_sha256=<hash>", "list voices", "speak hello"],
}

RISK_EXPLANATIONS = {
    "READ_ONLY": "can run automatically",
    "LOCAL_SAFE": "local change, allowed by default",
    "PERSONAL_DATA": "requires approval because it may expose private context",
    "EXTERNAL_SIDE_EFFECT": "requires approval because it can affect the outside world",
    "HIGH_RISK": "requires approval because it can change or control important state",
}

MESSAGE_LIVE_PROOF_NOTE = (
    "Live proof: Telegram, Instagram, iMessage, and KakaoTalk sends have recipient-confirmed V3 "
    "evidence. KakaoTalk required operator chat verification plus recipient confirmation after an "
    "outcome-unknown GUI attempt. Every new send remains approval-gated. Calls are disabled for "
    "the August 15 functional preview; any later re-enable and live proof requires a separate, "
    "operator-present approval gate. Trust the Capability Cockpit "
    "and `channel health` for the live delivery state."
)

EVERYDAY_CAPABILITY_SECTIONS = [
    (
        "Info",
        "weather, news, air quality, sunrise/sunset, history, holidays, Wikipedia, definitions",
        "weather in Tokyo",
    ),
    (
        "Productivity",
        "calendar, availability, email, tasks, reminders, timers",
        "am I free tomorrow afternoon?",
    ),
    (
        "Messages & calls",
        "send messages on KakaoTalk, Instagram, iMessage, or Telegram with approval; call routes exist but are disabled for the August 15 preview",
        "send 가상연락처일 a kakao saying on my way",
    ),
    (
        "Utilities",
        "translate, currency, unit conversion, time zones/time differences, relative dates, countdowns, calculate/BMI, spelling, password/UUID generation, text transforms/counts",
        "convert 100 USD to KRW",
    ),
    (
        "Markets",
        "crypto and stock price lookups",
        "stock price of AAPL",
    ),
    (
        "Research & Web",
        "web lookup, web search, page fetch/summarize, and research synthesis",
        "research the best setup for local AI notes",
    ),
    (
        "Writing",
        "paste_text, human_write, compose_and_write",
        "write a paragraph about Jarvis and type it",
    ),
    (
        "Fun",
        "jokes and lightweight playful prompts",
        "tell me a quick joke",
    ),
    (
        "Voice & images",
        "talk by voice (the always-on-top Jarvis HUD window with WhisperFlow dictation, push-to-talk, or a Telegram voice note), read text from a photo/document (on-device OCR), look up a contact's number",
        "what does this receipt say",
    ),
    (
        "Control plane",
        "capability cockpit, earned-trust checklist, channel health, Jarvis status, readiness report, safety/risk review, completion gate, and internal worker readiness",
        "cockpit summary",
    ),
]

EVERYDAY_FOCUS = {
    "info": {
        "name": "Info",
        "abilities": "weather, news, air quality, sunrise/sunset, history, holidays, Wikipedia, definitions",
        "example": "weather in Tokyo",
        "tools": {
            "get_weather",
            "get_news",
            "get_air_quality",
            "get_sun_times",
            "on_this_day",
            "next_holidays",
            "wiki_summary",
            "define",
        },
        "commands": [
            "weather in Tokyo",
            "Tokyo time",
            "current time Seoul",
            "news about Korea",
            "Korea news",
            "AI news today",
            "air quality in Seoul",
            "sunrise in Tokyo",
            "this day in history",
            "today in history",
            "next holidays in Korea",
            "tell me about Ada Lovelace",
            "define serendipity",
            "dictionary serendipity",
            "look up the definition of serendipity",
        ],
    },
    "productivity": {
        "name": "Productivity",
        "abilities": "calendar, availability, email, tasks, reminders, timers, Apple Reminders app",
        "example": "am I free tomorrow afternoon?",
        "tools": {
            "list_calendars",
            "list_events",
            "check_availability",
            "create_event",
            "update_event",
            "delete_event",
            "search_emails",
            "read_emails",
            "send_email",
            "read_email_body",
            "create_reminder",
            "set_reminder",
            "list_reminders",
            "cancel_reminders",
            "location_reminder_draft",
            "apple_reminders",
            "list_tasks",
            "search_tasks",
            "overdue_tasks",
            "next_task",
        },
        "commands": [
            "what's on my calendar today",
            "what is on my schedule today",
            "what am I doing today",
            "schedule today",
            "am I free tomorrow afternoon",
            "free today",
            "read recent emails",
            "what emails did Sam send me",
            "tasks due today",
            "what's due tomorrow",
            "today's tasks",
            "completed tasks",
            "overdue tasks",
            "urgent tasks",
            "todo board",
            "show next task",
            "what is my next task",
            "show alarms",
            "remind me to review Jarvis at 5pm",
            "set a reminder to call Sam at 5pm",
            "remind me to call Sam when I get home",
            "set a timer for 10 minutes",
            "timer 10m",
            "alarm 7am",
            "schedule lunch tomorrow at noon",
            "what's on my apple reminders",
        ],
    },
    "messages": {
        "name": "Messages & calls",
        "abilities": "send messages on KakaoTalk, Instagram, iMessage, Telegram, or email with approval; call routes exist but are disabled for the August 15 preview; look up contact handles before acting",
        "example": "send 가상연락처일 a kakao saying on my way",
        "tools": {
            "send_kakao",
            "send_instagram_dm",
            "send_imessage",
            "send_telegram",
            "send_email",
            "read_recent_imessages",
            "find_contact",
            "call_contact",
            "call_kakao",
            "call_telegram",
            "call_instagram",
        },
        "commands": [
            "channel health",
            "find contact 가상연락처일",
            "send 가상연락처일 a kakao saying on my way",
            "send 가상연락처일 a text saying running late",
            "send 가상연락처일 a telegram saying hi",
            "send 가상연락처이 a telegram saying 안녕하세요",
            "send mom an instagram dm saying hello",
            "send email to sam@example.com subject Hi body Hello",
            "call 가상연락처일",
            "facetime 가상연락처일",
            "call 가상연락처일 on kakao",
            "call mom on instagram",
        ],
    },
    "voice": {
        "name": "Voice & images",
        "abilities": "inspect voice readiness, use push-to-talk or Telegram voice-note transcripts with confirmation receipts, preview spoken replies, stop speech/listening, transcribe approved audio files, and run on-device OCR for photos/documents",
        "example": "voice command cockpit",
        "tools": {
            "voice_setup_check",
            "voice_capture_privacy_packet",
            "voice_command_cockpit",
            "voice_transcript_review",
            "voice_confirmation_packet",
            "voice_confirmation_receipt",
            "voice_route_gate_packet",
            "voice_route_proof_bundle",
            "voice_runtime_bridge_packet",
            "voice_audio_file_gate_packet",
            "voice_audio_file_transcription_preview",
            "voice_file_transcription_plan",
            "voice_input_plan",
            "voice_reply_preview",
            "voice_stop_intent_packet",
            "list_voices",
            "photo_document_intake_plan",
            "ocr_image",
        },
        "commands": [
            "voice setup check",
            "voice command cockpit",
            "voice capture privacy",
            "voice input plan push-to-talk",
            "voice transcript review: run command python3 --version",
            "voice confirmation: remember that Jarvis should ask before acting on speech",
            "voice confirmation receipt: run command python3 --version confirmed=true",
            "voice route gate: run command python3 --version confirmed=true",
            "voice reply preview: Jarvis is ready",
            "voice stop intent: stop listening",
            "voice file transcription plan: /path/to/audio.m4a",
            "voice audio file gate: /path/to/audio.m4a consent=true",
            "list voices",
            "photo intake plan: /path/to/receipt.png",
            "ocr image: /path/to/receipt.png",
            "what does this receipt say",
        ],
    },
    "utilities": {
        "name": "Utilities",
        "abilities": "translate, currency, unit conversion, time zones/time differences, relative dates, countdowns, calculate/BMI, spelling, password/UUID generation, text transforms/counts",
        "example": "convert 100 USD to KRW",
        "tools": {
            "current_time",
            "time_difference",
            "relative_date",
            "translate",
            "convert_currency",
            "convert_units",
            "days_until",
            "calculate",
            "calculate_bmi",
            "generate_password",
            "generate_uuid",
            "spell_word",
            "count_text",
            "transform_text",
        },
        "commands": [
            "translate hello to Korean",
            "Korean for good morning",
            "English for 사랑",
            "convert 100 USD to KRW",
            "usd to krw",
            "100 usd krw",
            "convert 10 km to miles",
            "what is 10 pounds in kg",
            "how many kg is 10 pounds",
            "how many cups in a liter",
            "Tokyo time",
            "what timezone am I in",
            "time difference between Seoul and London",
            "what date is tomorrow",
            "what date is in two days",
            "what day is next Friday",
            "days until Christmas",
            "days until Thanksgiving",
            "how many weeks until Christmas",
            "countdown Halloween",
            "when is Christmas",
            "when is MLK Day",
            "when is Thanksgiving",
            "calculate 18% of 240",
            "percentage change from 50 to 60",
            "total with 18% tip on 240",
            "total with 8% tax on 100",
            "split bill 240 3 ways",
            "bmi 70 kg 180 cm",
            "generate a 16 character password",
            "generate a 16 character password without symbols",
            "generate uuid",
            "add 15 and 20",
            "spell restaurant",
            "word count hello world",
            "letter count restaurant",
            "uppercase hello world",
            "uppercase the phrase hello world",
            "convert hello world to snake case",
            "camel case hello world",
            "initials the operator",
            "what are the initials of the operator",
            "slugify hello world",
            "repeat hello 3 times",
        ],
    },
    "markets": {
        "name": "Markets",
        "abilities": "crypto and stock price lookups",
        "example": "stock price of AAPL",
        "tools": {
            "get_crypto_price",
            "get_stock_price",
            "get_markets_overview",
        },
        "commands": [
            "how are the markets",
            "markets overview in Korean",
            "stocks today",
            "what is bitcoin worth",
            "crypto price",
            "stock price of AAPL",
            "AAPL price",
        ],
    },
    "research": {
        "name": "Research & Web",
        "abilities": "web lookup, web search, page fetch/summarize, and research synthesis",
        "example": "research the best setup for local AI notes",
        "tools": {
            "web_lookup",
            "research",
            "fetch_page",
            "web_search",
            "recent_browser_pages",
            "summarize_page",
        },
        "commands": [
            "research Zoey OS",
            "research local-first AI note taking",
            "web lookup OpenClaw agent safety",
            "look up OpenAI model context windows",
            "web search for Jarvis agent safety patterns",
            "fetch page https://example.com",
            "recent browser pages",
            "summarize latest page",
        ],
    },
    "writing": {
        "name": "Writing",
        "abilities": "paste_text, human_write, compose_and_write",
        "example": "write a paragraph about Jarvis and type it",
        "tools": {
            "paste_text",
            "human_write",
            "compose_and_write",
        },
        "commands": [
            "paste text: hello from Jarvis",
            "human write: hello from Jarvis",
            "write a paragraph about Jarvis and type it",
            "compose a quick project intro in google docs",
        ],
    },
    "fun": {
        "name": "Fun",
        "abilities": "jokes, coin flips, dice rolls, random numbers, option picks, and lightweight playful prompts",
        "example": "tell me a quick joke",
        "tools": {
            "tell_joke",
            "flip_coin",
            "roll_dice",
            "random_number",
            "choose_option",
        },
        "commands": [
            "joke",
            "tell me a joke",
            "flip a coin",
            "toss a coin",
            "roll a die",
            "roll a pair of dice",
            "roll 2 dice",
            "roll 2d6",
            "pick a random number",
            "pick a number between one and ten",
            "choose between pizza and sushi",
            "help me decide between pizza and sushi",
            "dad joke",
            "make me laugh",
        ],
    },
    "control_plane": {
        "name": "Control plane",
        "abilities": "read-only cockpit/status views for capabilities, earned-trust checklist, channel health, readiness, safety/risk, completion-claim blockers, and internal worker readiness",
        "example": "cockpit summary",
        "tools": {
            "capability_map",
            "capability_cockpit",
            "channel_health",
            "jarvis_status",
            "status_dashboard",
            "readiness_report",
            "safety_status",
            "risk_matrix",
            "completion_claim_gate",
            "subagent_fleet_status",
        },
        "commands": [
            "capability map",
            "capability cockpit",
            "cockpit summary",
            "trust checklist",
            "earned trust checklist",
            "why should I trust Jarvis",
            "channel health",
            "jarvis status",
            "status dashboard",
            "readiness report",
            "safety status",
            "risk matrix",
            "completion claim gate",
            "subagent fleet status",
            "internal orchestration",
        ],
    },
}


def describe_capabilities() -> str:
    lines = [
        "Everyday abilities:",
        "Ask naturally; Jarvis will choose the safest tool path and ask before private data, computer control, or outside-world actions.",
    ]
    for name, abilities, example in EVERYDAY_CAPABILITY_SECTIONS:
        lines.append(f"- {name}: {abilities}. Example: `{example}`")
    lines.append(f"- {MESSAGE_LIVE_PROOF_NOTE}")
    return "\n".join(lines)


def _bounded_int(value: Any, default: int, low: int = 1, high: int = 200) -> int:
    if isinstance(value, bool):
        return default
    try:
        number = int(value)
    except (TypeError, ValueError):
        number = default
    return max(low, min(high, number))


def _raw_limit_metadata(value: Any, *, limit: int) -> dict[str, Any]:
    if value is None:
        return {"limit": limit}
    if isinstance(value, bool):
        return {"limit": limit, "raw_limit": str(value)}
    try:
        int(value)
    except (TypeError, ValueError):
        return {"limit": limit, "raw_limit": _short_metadata(value, 80)}
    return {"limit": limit}


def _short(value: Any, limit: int = 500) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


def _short_metadata(value: Any, limit: int = 80) -> str:
    return LOCAL_PATH_RE.sub("<local-path>", _short(value, limit))


def _safe_text(value: Any, limit: int = 500) -> str:
    return LOCAL_PATH_RE.sub("<local-path>", _short(value, limit))


def _raw_registry_values_from_bound_list(list_tools: Callable[[], list[Any]]) -> tuple[list[Any], bool]:
    registry = getattr(list_tools, "__self__", None)
    raw_tools = getattr(registry, "_tools", None)
    if not isinstance(raw_tools, dict):
        return [], False
    try:
        return list(raw_tools.values()), True
    except Exception:
        return [], False


def _registry_tools_with_hidden_count(list_tools: Callable[[], list[Any]]) -> tuple[list[Any], int]:
    """Best-effort registry view; malformed tools must stay visible as hidden count."""
    unreadable = 0
    tools, recovered = _raw_registry_values_from_bound_list(list_tools)
    if not recovered:
        try:
            tools = list_tools()
        except Exception:
            return [], 1
    try:
        iterable = list(tools or [])
    except Exception:
        return [], 1

    readable: list[tuple[str, Any]] = []
    for tool in iterable:
        try:
            name = str(tool.name)
            if not name or LOCAL_PATH_RE.search(name):
                raise ValueError("unreadable tool name")
            str(tool.toolset)
            str(tool.description)
            str(tool.risk.name)
        except Exception:
            unreadable += 1
            continue
        readable.append((name, tool))
    return [tool for _, tool in sorted(readable, key=lambda item: item[0])], unreadable


def _normalize_focus(value: Any) -> str:
    focus = _safe_text(value, 160).lower()
    focus = re.sub(r"^(?:with|for|about|on)\s+", "", focus).strip()
    aliases = {
        "calendar": "productivity",
        "calendars": "productivity",
        "email": "productivity",
        "emails": "productivity",
        "gmail": "productivity",
        "mail": "productivity",
        "reminder": "productivity",
        "reminders": "productivity",
        "timer": "productivity",
        "timers": "productivity",
        "message": "messages",
        "messaging": "messages",
        "send": "messages",
        "sending": "messages",
        "text": "messages",
        "texts": "messages",
        "dm": "messages",
        "dms": "messages",
        "imessage": "messages",
        "imessages": "messages",
        "kakao": "messages",
        "kakaotalk": "messages",
        "telegram": "messages",
        "instagram": "messages",
        "call": "messages",
        "calls": "messages",
        "calling": "messages",
        "phone": "messages",
        "phone call": "messages",
        "phone calls": "messages",
        "facetime": "messages",
        "contact lookup": "messages",
        "contact lookups": "messages",
        "메시지": "messages",
        "메세지": "messages",
        "문자": "messages",
        "전송": "messages",
        "카카오": "messages",
        "카톡": "messages",
        "텔레그램": "messages",
        "아이메시지": "messages",
        "인스타그램": "messages",
        "인스타": "messages",
        "전화": "messages",
        "통화": "messages",
        "연락처": "messages",
        "speech": "voice",
        "mic": "voice",
        "microphone": "voice",
        "voice note": "voice",
        "voice notes": "voice",
        "audio": "voice",
        "photo": "voice",
        "photos": "voice",
        "image": "voice",
        "images": "voice",
        "ocr": "voice",
        "음성": "voice",
        "목소리": "voice",
        "마이크": "voice",
        "보이스": "voice",
        "사진": "voice",
        "이미지": "voice",
        "weather": "info",
        "news": "info",
        "info": "info",
        "wikipedia": "info",
        "definition": "info",
        "definitions": "info",
        "translation": "utilities",
        "translate": "utilities",
        "currency": "utilities",
        "conversion": "utilities",
        "conversions": "utilities",
        "utility": "utilities",
        "utilities": "utilities",
        "market": "markets",
        "markets": "markets",
        "stocks": "markets",
        "stock": "markets",
        "crypto": "markets",
        "research": "research",
        "web lookup": "research",
        "web search": "research",
        "browser research": "research",
        "browser": "research",
        "browse": "research",
        "browsing": "research",
        "web": "research",
        "internet": "research",
        "internet search": "research",
        "search": "research",
        "write": "writing",
        "writing": "writing",
        "drafting": "writing",
        "compose": "writing",
        "jokes": "fun",
        "joke": "fun",
        "control": "control_plane",
        "control plane": "control_plane",
        "control-plane": "control_plane",
        "cockpit": "control_plane",
        "capability cockpit": "control_plane",
        "capabilities cockpit": "control_plane",
        "status": "control_plane",
        "jarvis status": "control_plane",
        "health": "control_plane",
        "channel health": "control_plane",
        "what broke": "control_plane",
        "readiness": "control_plane",
        "readiness report": "control_plane",
        "completion": "control_plane",
        "completion status": "control_plane",
        "completion gate": "control_plane",
        "completion claim": "control_plane",
        "completion claim gate": "control_plane",
        "risk matrix": "control_plane",
        "orchestration": "control_plane",
        "internal orchestration": "control_plane",
        "agent status": "control_plane",
        "subagent status": "control_plane",
        "subagent fleet": "control_plane",
        "worker status": "control_plane",
        "task": "tasks",
        "todo": "tasks",
        "todos": "tasks",
        "note": "notes",
        "obsidian": "notes",
        "file": "files",
        "approval": "approvals",
        "approvals": "approvals",
        "safety": "safety",
        "security": "safety",
        "schedule": "scheduler",
        "schedules": "scheduler",
        "scheduled job": "scheduler",
        "scheduled jobs": "scheduler",
        "automation": "scheduler",
        "automations": "scheduler",
        "assistant automation": "scheduler",
        "assistant automations": "scheduler",
        "job": "scheduler",
        "jobs": "scheduler",
        "goal": "goals",
        "project": "goals",
        "projects": "goals",
        "decision": "decisions",
        "preference": "preferences",
        "pref": "preferences",
        "prefs": "preferences",
        "skill": "skills",
        "profile": "profile",
        "profiles": "profile",
        "person": "people",
        "contact": "people",
        "contacts": "people",
        "conversation": "conversation",
        "conversations": "conversation",
        "chat": "conversation",
        "chats": "conversation",
        "캘린더": "productivity",
        "일정": "productivity",
        "이메일": "productivity",
        "메일": "productivity",
        "리마인더": "productivity",
        "타이머": "productivity",
        "할일": "tasks",
        "투두": "tasks",
        "날씨": "info",
        "뉴스": "info",
        "정보": "info",
        "시장": "markets",
        "마켓": "markets",
        "주식": "markets",
        "코인": "markets",
        "암호화폐": "markets",
        "번역": "utilities",
        "환율": "utilities",
        "유틸": "utilities",
        "유틸리티": "utilities",
        "검색": "research",
        "웹검색": "research",
        "웹 검색": "research",
        "웹": "research",
        "인터넷": "research",
        "조사": "research",
        "리서치": "research",
        "글쓰기": "writing",
        "작성": "writing",
        "메모": "notes",
        "노트": "notes",
        "기억": "memory",
        "메모리": "memory",
        "승인": "approvals",
        "안전": "safety",
        "콕핏": "control_plane",
        "상태": "control_plane",
        "헬스": "control_plane",
        "채널상태": "control_plane",
        "채널 상태": "control_plane",
        "준비상태": "control_plane",
        "준비 상태": "control_plane",
        "오케스트레이션": "control_plane",
        "내부오케스트레이션": "control_plane",
        "내부 오케스트레이션": "control_plane",
        "에이전트상태": "control_plane",
        "에이전트 상태": "control_plane",
        "워커상태": "control_plane",
        "워커 상태": "control_plane",
    }
    return aliases.get(focus, focus)


def _safe_metadata(**extra: Any) -> dict[str, Any]:
    metadata = {
        "calls_model": False,
        "executes_tools": False,
        "queues_approval": False,
        "authorizes_execution": False,
        "authorizes_completion_claim": False,
        "approval_granted": False,
        "approves_request": False,
        "dismisses_request": False,
        "controls_computer": False,
        "reads_private_data": False,
        "writes_files": False,
        "writes_notes": False,
        "writes_memory": False,
        "external_side_effect": False,
        "requires_approval": False,
    }
    metadata.update(extra)
    return metadata


def _inspection_boundary() -> dict[str, bool]:
    return {
        "read_only": True,
        "calls_model": False,
        "calls_external_service": False,
        "executes_tools": False,
        "reads_private_data": False,
        "writes_files": False,
        "writes_notes": False,
        "writes_memory": False,
        "queues_approval": False,
        "authorizes_execution": False,
        "authorizes_completion_claim": False,
        "approval_granted": False,
        "approves_request": False,
        "dismisses_request": False,
        "controls_computer": False,
        "external_side_effect": False,
        "requires_approval": False,
    }


def _inspection_handoff(
    *,
    source: str,
    status: str,
    reason: str = "",
    next_commands: list[str] | None = None,
    **extra: Any,
) -> dict[str, Any]:
    return {
        "source": source,
        "status": status,
        "reason": reason,
        "next_commands": next_commands or ["capability map", "tool search: approvals", "risk matrix"],
        "changed": [],
        "state_changed": False,
        "content_in_handoff": False,
        "ready_for_operator": True,
        "boundary": _inspection_boundary(),
        **extra,
    }


def make_capability_tools(list_tools: Callable[[], list[Any]]):
    def _safe_command_list(commands: list[str]) -> str:
        return " | ".join(_safe_text(command, 500) for command in commands)

    def _capability_boundary() -> dict[str, bool]:
        return {
            "read_only": True,
            "calls_model": False,
            "calls_external_service": False,
            "executes_tools": False,
            "reads_private_data": False,
            "writes_files": False,
            "writes_notes": False,
            "writes_memory": False,
            "queues_approval": False,
            "authorizes_execution": False,
            "authorizes_completion_claim": False,
            "approval_granted": False,
            "approves_request": False,
            "dismisses_request": False,
            "controls_computer": False,
            "external_side_effect": False,
            "requires_approval": False,
        }

    def _capability_handoff(
        *,
        focus: str,
        status: str,
        by_toolset: dict[str, list[Any]] | None = None,
        tools: list[Any] | None = None,
        risk_counts: Counter[str] | None = None,
        unreadable_registry_tools: int = 0,
    ) -> dict[str, Any]:
        by_toolset = by_toolset or {}
        tools = tools or []
        risk_counts = risk_counts or Counter()
        section_rows: list[dict[str, Any]] = []
        for toolset in sorted(by_toolset):
            group = sorted(by_toolset[toolset], key=lambda tool: (tool.risk, tool.name))
            label = TOOLSET_LABELS.get(toolset, toolset.replace("_", " "))
            section_rows.append(
                {
                    "toolset": _safe_text(toolset, 120),
                    "label": _safe_text(label, 160),
                    "tool_count": len(group),
                    "risk_counts": dict(sorted(Counter(tool.risk.name for tool in group).items())),
                    "example_tools": [_safe_text(tool.name, 120) for tool in _example_tools(toolset, group)[:4]],
                }
            )
        return {
            "source": "capability_map",
            "status": status,
            "focus": focus,
            "toolsets": len(by_toolset),
            "tools": len(tools),
            "unreadable_registry_tools": unreadable_registry_tools,
            "risk_counts": dict(sorted(risk_counts.items())),
            "section_rows": section_rows,
            "next_commands": [
                "capability map",
                "tool search: approvals",
                "tool detail: run_shell_command",
                "risk matrix",
                "help safety",
            ],
            "changed": [],
            "state_changed": False,
            "content_in_handoff": False,
            "ready_for_operator": True,
            "boundary": _capability_boundary(),
        }

    def _example_tools(toolset: str, group: list[Any]) -> list[Any]:
        preferred = {
            "core": [
                "capability_map",
                "tool_search",
                "tool_detail",
                "risk_matrix",
                "priority_goal",
                "harness_status",
                "harness_completion_assessment",
                "evidence_ledger",
                "completion_claim_gate",
                "completion_next_proof_packet",
                "completion_proof_refresh_packet",
                "harness_cycle_preview",
                "harness_lifecycle_state",
                "harness_control_surface",
                "harness_operations_brief",
                "execution_mission_control",
                "save_execution_case",
                "inspect_execution_case",
                "append_execution_case_evidence",
                "execution_case_gate",
                "execution_case_review_packet",
                "execution_case_closure_packet",
                "execution_case_timeline",
                "agi_gate_report",
                "agi_next_build_move",
                "architecture_map",
                "roadmap_report",
                "model_routing_status",
                "specialist_router_contract",
                "specialist_orchestration_packet",
                "specialist_route_quality",
                "specialist_execution_readiness",
                "specialist_handoff_receipt",
                "specialist_handoff_quality_gate",
                "specialist_proposal_gate",
                "specialist_action_proposal_contract",
                "specialist_tool_dry_run_packet",
                "specialist_proposal_completion_gate",
                "specialist_execution_handoff_packet",
                "specialist_post_run_closure_packet",
                "specialist_cycle_ledger",
            ],
            "safety": [
                "safety_status",
                "readiness_report",
                "autonomy_plan",
                "agent_loop_packet",
                "risky_request_lifecycle",
                "execution_acceptance_gate",
                "execution_mission_control",
                "save_execution_case",
                "inspect_execution_case",
                "append_execution_case_evidence",
                "execution_case_gate",
                "execution_case_review_packet",
                "execution_case_closure_packet",
                "execution_case_timeline",
                "execution_readiness_matrix",
                "dispatch_decision_packet",
                "command_intake_packet",
                "execution_governor_packet",
                "planner_gap_packet",
                "risk_preflight",
                "action_rehearsal",
            ],
            "continuity": [
                "harness_build_slice",
                "continuation_packet",
                "autonomy_cycle_ledger",
                "build_target_packet",
                "priority_stack",
                "next_action_packet",
                "work_queue",
                "focus_brief",
                "work_session_packet",
                "checkpoint_recovery_preview",
                "checkpoint_recovery_receipt",
            ],
            "personal": [
                "legacy_connector_migration_audit",
                "integration_proof_bundle",
                "integration_implementation_review",
                "integration_preflight_contract",
                "integration_route_lock",
                "integration_execution_matrix",
                "integration_adapter_manifest",
                "integration_adapter_probe",
                "integration_adapter_acceptance",
                "integration_metadata_preview",
                "integration_implementation_spec",
                "integration_rehearsal_receipt",
                "integration_enablement_gate",
                "integration_promotion_gate",
                "integration_runbook",
                "integration_dry_run_contract",
                "integration_scope_packet",
                "integration_action_preview",
                "integration_migration_plan",
                "integration_boundary_contract",
                "integration_status",
            ],
        }.get(toolset, [])
        by_name = {tool.name: tool for tool in group}
        examples = [by_name[name] for name in preferred if name in by_name]
        examples.extend(tool for tool in group if tool.name not in preferred)
        return examples

    def capability_map(args: dict[str, Any]) -> ToolResult:
        focus = _normalize_focus(args.get("focus"))
        everyday_focus = EVERYDAY_FOCUS.get(focus)
        tools, unreadable_registry_tools = _registry_tools_with_hidden_count(list_tools)
        if everyday_focus:
            names = everyday_focus["tools"]
            tools = [tool for tool in tools if tool.name in names]
        elif focus:
            tools = [
                tool
                for tool in tools
                if focus in tool.toolset.lower()
                or focus in tool.name.lower()
                or focus in tool.description.lower()
                or focus in TOOLSET_LABELS.get(tool.toolset, tool.toolset).lower()
            ]
            if not tools:
                handoff = _capability_handoff(
                    focus=focus,
                    status="empty",
                    unreadable_registry_tools=unreadable_registry_tools,
                )
                note = ""
                if unreadable_registry_tools:
                    note = f" {unreadable_registry_tools} unreadable registry tool(s) hidden for safety."
                return ToolResult(
                    "capability_map",
                    True,
                    f"No Jarvis capabilities matched '{focus}'. Try `capability map`.{note}",
                    _safe_metadata(
                        toolsets=0,
                        tools=0,
                        focus=focus,
                        unreadable_registry_tools=unreadable_registry_tools,
                        capability_map_handoff_ready=True,
                        capability_map_handoff=handoff,
                    ),
                )

        by_toolset: dict[str, list[Any]] = defaultdict(list)
        risk_counts: Counter[str] = Counter()
        for tool in tools:
            by_toolset[tool.toolset].append(tool)
            risk_counts[tool.risk.name] += 1

        lines = ["Jarvis capability map:"]
        if focus:
            lines.append(f"Focus: {focus}")
        if everyday_focus:
            matching_tool_names = ", ".join(sorted(tool.name for tool in tools)) or "none"
            lines.extend(
                [
                    "",
                    f"Everyday ability: {everyday_focus['name']}",
                    f"- Includes: {everyday_focus['abilities']}.",
                    f"- Example: `{everyday_focus['example']}`",
                    f"- Matching tools: {matching_tool_names}",
                ]
            )
            if focus == "messages":
                lines.append(f"- {MESSAGE_LIVE_PROOF_NOTE}")
        if not focus or focus in {"personal", "everyday", "abilities", "capabilities"}:
            lines.extend(["", describe_capabilities()])

        lines.extend(["", "Safety model:"])
        for risk_name, explanation in RISK_EXPLANATIONS.items():
            count = risk_counts.get(risk_name, 0)
            if count:
                lines.append(f"- {risk_name}: {count} tool(s), {explanation}.")

        lines.extend(["", "Capability areas:"])
        for toolset in sorted(by_toolset):
            group = sorted(by_toolset[toolset], key=lambda tool: (tool.risk, tool.name))
            example_group = _example_tools(toolset, group)
            label = TOOLSET_LABELS.get(toolset, toolset.replace("_", " "))
            risks = ", ".join(f"{risk}={count}" for risk, count in sorted(Counter(tool.risk.name for tool in group).items()))
            examples = ", ".join(tool.name for tool in example_group[:4])
            extra = f" (+{len(group) - 4} more)" if len(group) > 4 else ""
            lines.append(f"- {label}: {len(group)} tool(s), {risks}. Examples: {examples}{extra}.")

        if unreadable_registry_tools:
            lines.append("")
            lines.append(f"Note: {unreadable_registry_tools} unreadable registry tool(s) hidden for safety.")

        lines.extend(["", "Good starting commands:"])
        command_groups = STARTER_COMMANDS.items()
        if everyday_focus:
            command_groups = [(everyday_focus["name"].lower(), everyday_focus["commands"])]
        elif focus:
            command_groups = [(name, commands) for name, commands in STARTER_COMMANDS.items() if name in by_toolset or focus in name]
        for name, commands in command_groups:
            label = TOOLSET_LABELS.get(name, name)
            lines.append(f"- {label}: " + _safe_command_list(commands))

        lines.extend(
            [
                "",
                "Rule: this only maps Jarvis capabilities. It does not run tools, approve requests, read private data, control the computer, or queue approvals.",
                "Rule of thumb: Jarvis may explain, search, summarize, plan, and organize local memory by default. It asks first before computer control, private data access, shell/code execution, destructive changes, reminders, or outside-world actions.",
            ]
        )
        return ToolResult(
            "capability_map",
            True,
            "\n".join(lines),
            _safe_metadata(
                toolsets=len(by_toolset),
                tools=len(tools),
                focus=focus,
                unreadable_registry_tools=unreadable_registry_tools,
                capability_map_handoff_ready=True,
                capability_map_handoff=_capability_handoff(
                    focus=focus,
                    status="ok",
                    by_toolset=by_toolset,
                    tools=tools,
                    risk_counts=risk_counts,
                    unreadable_registry_tools=unreadable_registry_tools,
                ),
            ),
        )

    return capability_map


def make_tool_search_tool(list_tools: Callable[[], list[Any]]):
    def tool_search(args: dict[str, Any]) -> ToolResult:
        query = _safe_text(args.get("query"), 500)
        limit = _bounded_int(args.get("limit"), 12)
        if not query:
            return ToolResult(
                "tool_search",
                False,
                "Search query is required. Try `tool search: approvals`.",
                _safe_metadata(
                    reason="missing_query",
                    query_chars=0,
                    **_raw_limit_metadata(args.get("limit"), limit=limit),
                    tool_search_handoff_ready=True,
                    tool_search_handoff=_inspection_handoff(
                        source="tool_search",
                        status="refused",
                        reason="missing_query",
                        query="",
                        query_chars=0,
                        count=0,
                        limit=limit,
                        matches=[],
                        next_commands=["tool search: approvals", "capability map", "risk matrix"],
                    ),
                ),
            )
        terms = [term.lower() for term in query.split() if term.strip()]
        tools, unreadable_registry_tools = _registry_tools_with_hidden_count(list_tools)
        matches = []
        for tool in tools:
            label = TOOLSET_LABELS.get(tool.toolset, tool.toolset.replace("_", " "))
            haystack = " ".join([tool.name, tool.description, tool.toolset, label, tool.risk.name]).lower()
            if all(term in haystack for term in terms):
                matches.append(tool)
        matches = matches[:limit]
        if not matches:
            lines = [f"No Jarvis tools matched '{query}'. Try `capability map` for a broader view."]
            if unreadable_registry_tools:
                lines.append(f"{unreadable_registry_tools} unreadable registry tool(s) hidden for safety.")
            return ToolResult(
                "tool_search",
                True,
                " ".join(lines),
                _safe_metadata(
                    query=query,
                    query_chars=len(query),
                    count=0,
                    matches=[],
                    limit=limit,
                    unreadable_registry_tools=unreadable_registry_tools,
                    tool_search_handoff_ready=True,
                    tool_search_handoff=_inspection_handoff(
                        source="tool_search",
                        status="empty",
                        query=query,
                        query_chars=len(query),
                        count=0,
                        limit=limit,
                        matches=[],
                        unreadable_registry_tools=unreadable_registry_tools,
                        next_commands=["capability map", "tool search: approvals", "risk matrix"],
                    ),
                ),
            )
        lines = [f"Jarvis tool search: {query}"]
        structured = []
        for tool in matches:
            risk_note = RISK_EXPLANATIONS.get(tool.risk.name, "review before using")
            label = TOOLSET_LABELS.get(tool.toolset, tool.toolset.replace("_", " "))
            lines.append(f"- {tool.name} [{label}, {tool.risk.name}]: {tool.description} ({risk_note})")
            structured.append(
                {
                    "name": tool.name,
                    "toolset": tool.toolset,
                    "label": label,
                    "risk": tool.risk.name,
                    "description": tool.description,
                    "risk_note": risk_note,
                }
            )
        lines.extend(
            [
                "",
                "Rule: this only searches Jarvis capabilities. It does not run tools, approve requests, read private data, control the computer, or queue approvals.",
            ]
        )
        if unreadable_registry_tools:
            lines.extend(["", f"Note: {unreadable_registry_tools} unreadable registry tool(s) hidden for safety."])
        return ToolResult(
            "tool_search",
            True,
            "\n".join(lines),
            _safe_metadata(
                query=query,
                query_chars=len(query),
                count=len(matches),
                matches=structured,
                limit=limit,
                unreadable_registry_tools=unreadable_registry_tools,
                tool_search_handoff_ready=True,
                tool_search_handoff=_inspection_handoff(
                    source="tool_search",
                    status="ok",
                    query=query,
                    query_chars=len(query),
                    count=len(matches),
                    limit=limit,
                    matches=structured,
                    unreadable_registry_tools=unreadable_registry_tools,
                    next_commands=["tool detail: " + structured[0]["name"], "risk matrix", "capability map"],
                ),
            ),
        )

    return tool_search


def make_tool_detail_tool(list_tools: Callable[[], list[Any]]):
    def tool_detail(args: dict[str, Any]) -> ToolResult:
        raw_name = args.get("name") if args.get("name") is not None else args.get("tool")
        name = _safe_text(raw_name, 160)
        if not name:
            return ToolResult(
                "tool_detail",
                False,
                "Tool name is required. Try `tool detail: run_shell_command`.",
                _safe_metadata(
                    reason="missing_name",
                    found=False,
                    raw_name=_short_metadata(raw_name, 80),
                    tool_detail_handoff_ready=True,
                    tool_detail_handoff=_inspection_handoff(
                        source="tool_detail",
                        status="refused",
                        reason="missing_name",
                        name="",
                        found=False,
                        matches=[],
                        next_commands=["tool detail: run_shell_command", "tool search: approvals", "capability map"],
                    ),
                ),
            )
        tools, unreadable_registry_tools = _registry_tools_with_hidden_count(list_tools)
        by_name = {tool.name: tool for tool in tools}
        tool = by_name.get(name)
        if tool is None:
            lowered = name.lower()
            candidates = [tool for tool in tools if lowered in tool.name.lower()][:8]
            if not candidates:
                note = ""
                if unreadable_registry_tools:
                    note = f" {unreadable_registry_tools} unreadable registry tool(s) hidden for safety."
                return ToolResult(
                    "tool_detail",
                    True,
                    f"No Jarvis tool named '{name}'. Try `tool search: {name}`.{note}",
                    _safe_metadata(
                        name=name,
                        found=False,
                        matches=[],
                        unreadable_registry_tools=unreadable_registry_tools,
                        tool_detail_handoff_ready=True,
                        tool_detail_handoff=_inspection_handoff(
                            source="tool_detail",
                            status="empty",
                            name=name,
                            found=False,
                            matches=[],
                            unreadable_registry_tools=unreadable_registry_tools,
                            next_commands=[f"tool search: {name}", "capability map", "risk matrix"],
                        ),
                    ),
                )
            lines = [f"No exact Jarvis tool named '{name}'. Close matches:"]
            lines.extend(f"- {candidate.name} [{candidate.risk.name}]" for candidate in candidates)
            lines.append("")
            lines.append(f"Try: `tool detail: {candidates[0].name}`")
            if unreadable_registry_tools:
                lines.append(f"Note: {unreadable_registry_tools} unreadable registry tool(s) hidden for safety.")
            return ToolResult(
                "tool_detail",
                True,
                "\n".join(lines),
                _safe_metadata(
                    name=name,
                    found=False,
                    matches=[candidate.name for candidate in candidates],
                    unreadable_registry_tools=unreadable_registry_tools,
                    tool_detail_handoff_ready=True,
                    tool_detail_handoff=_inspection_handoff(
                        source="tool_detail",
                        status="partial",
                        name=name,
                        found=False,
                        matches=[candidate.name for candidate in candidates],
                        unreadable_registry_tools=unreadable_registry_tools,
                        next_commands=[f"tool detail: {candidates[0].name}", f"tool search: {name}", "capability map"],
                    ),
                ),
            )
        label = TOOLSET_LABELS.get(tool.toolset, tool.toolset.replace("_", " "))
        risk_note = RISK_EXPLANATIONS.get(tool.risk.name, "review before using")
        approval_required = tool.risk.name not in {"READ_ONLY", "LOCAL_SAFE"}
        auto_behavior = "can run automatically" if not approval_required else "must stop for explicit approval"
        lines = [
            f"Jarvis tool detail: {tool.name}",
            f"- area: {label}",
            f"- risk: {tool.risk.name}",
            f"- approval required: {'yes' if approval_required else 'no'}",
            f"- behavior: {auto_behavior}",
            f"- description: {tool.description}",
            f"- risk note: {risk_note}",
            "",
            "Rule: this only inspects tool metadata. It does not run the tool, approve requests, read private data, control the computer, or queue approvals.",
        ]
        if unreadable_registry_tools:
            lines.extend(["", f"Note: {unreadable_registry_tools} unreadable registry tool(s) hidden for safety."])
        return ToolResult(
            "tool_detail",
            True,
            "\n".join(lines),
            _safe_metadata(
                name=tool.name,
                found=True,
                toolset=tool.toolset,
                label=label,
                risk=tool.risk.name,
                approval_required=approval_required,
                description=tool.description,
                unreadable_registry_tools=unreadable_registry_tools,
                tool_detail_handoff_ready=True,
                tool_detail_handoff=_inspection_handoff(
                    source="tool_detail",
                    status="ok",
                    name=tool.name,
                    found=True,
                    toolset=tool.toolset,
                    label=label,
                    risk=tool.risk.name,
                    approval_required=approval_required,
                    description=tool.description,
                    risk_note=risk_note,
                    unreadable_registry_tools=unreadable_registry_tools,
                    next_commands=[f"tool search: {tool.toolset}", "risk matrix", "capability map"],
                ),
            ),
        )

    return tool_detail


def make_risk_matrix_tool(list_tools: Callable[[], list[Any]]):
    def risk_matrix(args: dict[str, Any]) -> ToolResult:
        limit = _bounded_int(args.get("limit"), 8, high=50)
        tools, unreadable_registry_tools = _registry_tools_with_hidden_count(list_tools)
        by_toolset: dict[str, list[Any]] = defaultdict(list)
        by_risk: dict[str, list[Any]] = defaultdict(list)
        for tool in tools:
            by_toolset[tool.toolset].append(tool)
            by_risk[tool.risk.name].append(tool)

        lines = ["Jarvis risk matrix:"]
        lines.append("")
        lines.append("Risk totals:")
        for risk_name in RISK_EXPLANATIONS:
            group = by_risk.get(risk_name, [])
            if group:
                lines.append(f"- {risk_name}: {len(group)} tool(s), {RISK_EXPLANATIONS[risk_name]}.")

        lines.append("")
        lines.append("Risk by area:")
        matrix: dict[str, dict[str, int]] = {}
        for toolset in sorted(by_toolset):
            group = by_toolset[toolset]
            counts = Counter(tool.risk.name for tool in group)
            matrix[toolset] = dict(counts)
            label = TOOLSET_LABELS.get(toolset, toolset.replace("_", " "))
            risk_bits = ", ".join(f"{risk}={counts.get(risk, 0)}" for risk in RISK_EXPLANATIONS if counts.get(risk, 0))
            lines.append(f"- {label}: {risk_bits}")

        risk_gated = [
            tool
            for tool in tools
            if tool.risk.name in {"PERSONAL_DATA", "EXTERNAL_SIDE_EFFECT", "HIGH_RISK"}
        ]
        if risk_gated:
            lines.append("")
            lines.append(f"Approval-gated examples ({min(limit, len(risk_gated))} shown):")
            for tool in sorted(risk_gated, key=lambda item: (item.risk.name, item.toolset, item.name))[:limit]:
                label = TOOLSET_LABELS.get(tool.toolset, tool.toolset.replace("_", " "))
                lines.append(f"- {tool.name} [{label}, {tool.risk.name}]")

        if unreadable_registry_tools:
            lines.append("")
            lines.append(f"Note: {unreadable_registry_tools} unreadable registry tool(s) hidden for safety.")

        lines.extend(
            [
                "",
                "Rule: read-only and local-safe tools may run automatically. Personal data, external effects, shell/code, destructive actions, and computer control stay approval-gated.",
            ]
        )
        return ToolResult(
            "risk_matrix",
            True,
            "\n".join(lines),
            _safe_metadata(
                tools=len(tools),
                risk_totals={risk: len(group) for risk, group in by_risk.items()},
                matrix=matrix,
                approval_gated=len(risk_gated),
                unreadable_registry_tools=unreadable_registry_tools,
                **_raw_limit_metadata(args.get("limit"), limit=limit),
                risk_matrix_handoff_ready=True,
                risk_matrix_handoff=_inspection_handoff(
                    source="risk_matrix",
                    status="ok",
                    tools=len(tools),
                    risk_totals={risk: len(group) for risk, group in by_risk.items()},
                    matrix=matrix,
                    approval_gated=len(risk_gated),
                    unreadable_registry_tools=unreadable_registry_tools,
                    limit=limit,
                    approval_gated_examples=[
                        {
                            "name": _safe_text(tool.name, 120),
                            "toolset": _safe_text(tool.toolset, 120),
                            "label": _safe_text(TOOLSET_LABELS.get(tool.toolset, tool.toolset.replace("_", " ")), 160),
                            "risk": tool.risk.name,
                        }
                        for tool in sorted(risk_gated, key=lambda item: (item.risk.name, item.toolset, item.name))[:limit]
                    ],
                    next_commands=["tool search: approval", "tool detail: run_shell_command", "capability map"],
                ),
            ),
        )

    return risk_matrix
