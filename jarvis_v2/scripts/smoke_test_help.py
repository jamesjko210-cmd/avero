from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from jarvis_v2.agent.planner import RuleBasedPlanner
from jarvis_v2.scripts.test_runtime import make_temp_runtime
from jarvis_v2.v3_commands import (
    V3_DASHBOARD_COMMAND,
    V3_DASHBOARD_INFO_COMMAND,
    V3_DIAGNOSE_PYTHON_COMMAND,
    V3_DIAGNOSE_PYTHON_JSON_COMMAND,
    V3_ONE_SHOT_TIME_COMMAND,
)


CASES = {
    "help": [
        "Jarvis help topics:",
        "Everyday abilities:",
        "Info: weather, news, air quality, sunrise/sunset, history, holidays, Wikipedia, definitions. Example: `weather in Tokyo`",
        "Productivity: calendar, availability, email, tasks, reminders, timers. Example: `am I free tomorrow afternoon?`",
        "Utilities: translate, currency, unit conversion, time zones/time differences, relative dates, countdowns, calculate/BMI, spelling, password/UUID generation, text transforms/counts. Example: `convert 100 USD to KRW`",
        "Markets: crypto and stock price lookups. Example: `stock price of AAPL`",
        "Research & Web: web lookup, web search, page fetch/summarize, and research synthesis. Example: `research the best setup for local AI notes`",
        "Writing: paste_text, human_write, compose_and_write. Example: `write a paragraph about Jarvis and type it`",
        "Fun: jokes and lightweight playful prompts. Example: `tell me a quick joke`",
        "## Core",
        "jarvis status",
        "priority goal",
        "## Control",
        "capability cockpit",
        "channel health",
        "trust checklist",
        "## Safety",
        "safety status",
        "prototype readiness",
        "## Approvals",
        "pending approvals",
    ],
    "help core": [
        "Jarvis help: core",
        "command diagnosis: run command python3 --version",
        f"terminal one-shot: {V3_DIAGNOSE_PYTHON_JSON_COMMAND}",
        f"terminal one-shot: {V3_ONE_SHOT_TIME_COMMAND}",
        f"terminal dashboard: {V3_DASHBOARD_COMMAND}",
        f"terminal dashboard help: {V3_DASHBOARD_INFO_COMMAND}",
    ],
    "help control": [
        "Jarvis help: control",
        "capability map",
        "capability cockpit",
        "cockpit",
        "channel health",
        "trust checklist",
        "earned trust checklist",
        "why should I trust Jarvis",
        "jarvis status",
        "status dashboard",
        "readiness report",
        "safety status",
        "risk matrix",
        "completion claim gate",
        "subagent fleet status",
        "internal orchestration",
        "agent status",
        "worker status",
        "visible control plane",
        "why a lane has earned trust",
        "for trust evidence, start with `trust checklist`",
        "READ_ONLY",
        "It does not approve, send, call, execute tools, read private data, control the computer, or change state.",
    ],
    "help cockpit": [
        "Jarvis help: control",
        "capability cockpit",
        "channel health",
        "completion claim gate",
        "READ_ONLY",
    ],
    "help status": [
        "Jarvis help: control",
        "jarvis status",
        "status dashboard",
        "subagent fleet status",
        "internal orchestration",
        "READ_ONLY",
    ],
    "help channel health": [
        "Jarvis help: control",
        "channel health",
        "visible control plane",
        "READ_ONLY",
    ],
    "help channel status": [
        "Jarvis help: control",
        "channel health",
        "visible control plane",
        "READ_ONLY",
    ],
    "help risk matrix": [
        "Jarvis help: control",
        "risk matrix",
        "READ_ONLY",
    ],
    "help contrl": [
        "Jarvis help: control",
        "capability cockpit",
        "channel health",
        "READ_ONLY",
    ],
    "help cokpit": [
        "Jarvis help: control",
        "capability cockpit",
        "channel health",
        "READ_ONLY",
    ],
    "help statuz": [
        "Jarvis help: control",
        "jarvis status",
        "status dashboard",
        "READ_ONLY",
    ],
    "help safety": [
        "Jarvis help: safety",
        "prototype readiness",
        "setup check",
        "storage status",
        "storage recovery plan",
        "storage recovery check",
        "voice setup check",
        "computer control status",
        "risk preflight",
        "command diagnosis: run command python3 --version",
        f"terminal diagnose: {V3_DIAGNOSE_PYTHON_COMMAND}",
        "approval readiness, approval packet, and approval chain proof reviewed; verification receipt linked",
        "Approval is rejected until approval readiness and the matching last-look approval packet have been viewed; approval chain proof then links approval",
    ],
    "help approvals": [
        "Jarvis help: approvals",
        "approval readiness <id|latest>",
        "approval packet <id|latest>",
        "approval chain proof <id|latest>",
        "Run approval readiness first, then the approval packet",
        "then approval chain proof",
        "Approve commands only rerun after readiness and the matching approval packet have been viewed.",
    ],
    "help approval queue": [
        "Jarvis help: approvals",
        "pending approvals",
        "approval readiness <id|latest>",
        "approval packet <id|latest>",
        "approval chain proof <id|latest>",
    ],
    "help pending approvals": [
        "Jarvis help: approvals",
        "pending approvals",
        "approval readiness <id|latest>",
        "approval packet <id|latest>",
        "approval chain proof <id|latest>",
    ],
    "help tasks": [
        "Jarvis help: tasks",
        "what's due tomorrow",
        "today's tasks",
        "completed tasks",
        "overdue tasks",
        "urgent tasks",
        "todo board",
        "show next task",
        "task completion packet 1",
        "complete task 1 with evidence: verified in recent tool runs",
    ],
    "jarvis help automation": [
        "Jarvis help: automation",
        "morning startup",
        "schedule assistant basics",
        "resume job State Snapshot",
        "resume job Conversation Compaction",
        "list scheduled jobs",
        "approval",
    ],
    "help morning brief": [
        "Jarvis help: automation",
        "schedule daily brief",
        "run morning brief now",
        "approval",
    ],
    "help daily brief": [
        "Jarvis help: automation",
        "schedule daily brief",
        "run morning brief now",
        "approval",
    ],
    "help calendar": [
        "Jarvis help: calendar",
        "what's on my calendar today",
        "what is on my schedule today",
        "what am I doing today",
        "schedule today",
        "free today",
        "schedule PTO all day tomorrow",
        "move event evt123 to June 20 at 2pm",
        "rename event evt123 to Lunch with Sam",
        "HIGH_RISK",
        "exact event id",
    ],
    "help translation": [
        "Jarvis help: translation",
        "translate hello to Korean",
        "how do you say thank you in Korean",
        "what is 사랑 in English",
        "LOCAL_SAFE",
        "sensitive personal data",
    ],
    "help markets": [
        "Jarvis help: markets",
        "what is bitcoin worth",
        "stock price of AAPL",
        "AAPL stock",
        "LOCAL_SAFE",
        "not financial advice",
    ],
    "help currency": [
        "Jarvis help: currency",
        "convert 100 USD to KRW",
        "100 dollars in won",
        "how much is 50 euros in dollars",
        "LOCAL_SAFE",
    ],
    "help weather": [
        "Jarvis help: weather",
        "what's the weather",
        "weather in Tokyo",
        "is it going to rain",
        "LOCAL_SAFE",
    ],
    "help news": [
        "Jarvis help: news",
        "what is the news",
        "headlines",
        "news about Korea",
        "LOCAL_SAFE",
    ],
    "help info": [
        "Jarvis help: info",
        "weather in Tokyo",
        "air quality in Seoul",
        "sunrise in Tokyo",
        "this day in history",
        "next holidays in Korea",
        "tell me about Ada Lovelace",
        "define serendipity",
        "dictionary serendipity",
        "look up the definition of serendipity",
        "LOCAL_SAFE",
    ],
    "help productivity": [
        "Jarvis help: productivity",
        "what's on my calendar today",
        "what am I doing today",
        "am I free tomorrow afternoon",
        "free today",
        "read recent emails",
        "what emails did Sam send me",
        "show alarms",
        "remind me to review Jarvis at 5pm",
        "set a reminder to call Sam at 5pm",
        "remind me to call Sam when I get home",
        "set a timer for 10 minutes",
        "timer 10m",
        "alarm 7am",
        "approval-aware",
    ],
    "help utilities": [
        "Jarvis help: utilities",
        "translate hello to Korean",
        "convert 100 USD to KRW",
        "convert 10 km to miles",
        "what time zone am I in",
        "time difference between Seoul and London",
        "what date is tomorrow",
        "what date is in two days",
        "what day is next Friday",
        "days until Christmas",
        "days until Thanksgiving",
        "when is MLK Day",
        "when is Thanksgiving",
        "calculate 18% of 240",
        "percentage change from 50 to 60",
        "total with 8% tax on 100",
        "bmi 70 kg 180 cm",
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
        "generate a 16 character password",
        "generate a 16 character password without symbols",
        "generate uuid",
        "repeat hello 3 times",
    ],
    "help research": [
        "Jarvis help: research",
        "research Zoey OS",
        "research local-first AI note taking",
        "web lookup OpenClaw agent safety",
        "look up OpenAI model context windows",
        "web search for Jarvis agent safety patterns",
        "fetch page https://example.com",
        "recent browser pages",
        "summarize latest page",
        "public sources",
        "does not send messages, approve requests, read private accounts, write notes, or change external state",
    ],
    "help writing": [
        "Jarvis help: writing",
        "paste text: hello from Jarvis",
        "human write: hello from Jarvis",
        "write a paragraph about Jarvis and type it",
        "compose a quick project intro in google docs",
        "HIGH_RISK",
    ],
    "help fun": [
        "Jarvis help: fun",
        "tell me a joke",
        "toss a coin",
        "roll a pair of dice",
        "roll 2 dice",
        "pick a random number",
        "pick a number between one and ten",
        "choose between pizza and sushi",
        "help me decide between pizza and sushi",
        "dad joke",
        "make me laugh",
        "read-only",
    ],
}


READ_ONLY_FLAGS = [
    "calls_model",
    "calls_external_service",
    "executes_tools",
    "reads_personal_data",
    "reads_private_data",
    "executes_side_effect",
    "external_side_effect",
    "writes_files",
    "writes_database",
    "writes_memory",
    "writes_notes",
    "queues_approval",
    "requires_approval",
    "controls_computer",
    "speaks",
    "completes_tasks",
    "authorizes_execution",
    "authorizes_completion_claim",
    "approval_granted",
]


def assert_help_aliases_route_to_help_or_capabilities() -> None:
    planner = RuleBasedPlanner()
    expected = {
        "show help": ("jarvis_help", {"topic": ""}),
        "show latest help": ("jarvis_help", {"topic": ""}),
        "jarvis help please": ("jarvis_help", {"topic": ""}),
        "help me": ("jarvis_help", {"topic": ""}),
        "help approvals please": ("jarvis_help", {"topic": "approvals"}),
        "how do I use Jarvis": ("jarvis_help", {"topic": ""}),
        "how do I use you": ("jarvis_help", {"topic": ""}),
        "what should I ask Jarvis": ("jarvis_help", {"topic": ""}),
        "what commands can I use": ("jarvis_help", {"topic": ""}),
        "what commands do you understand": ("jarvis_help", {"topic": ""}),
        "show me commands": ("jarvis_help", {"topic": ""}),
        "show commands": ("jarvis_help", {"topic": ""}),
        "command list please": ("jarvis_help", {"topic": ""}),
        # Real gap found live 2026-07-09: "what can you help with" fell
        # through to chat while "help", "jarvis help", and "show help" all
        # worked.
        "what can you help with": ("jarvis_help", {"topic": ""}),
        "help control": ("jarvis_help", {"topic": "control"}),
        "jarvis help control": ("jarvis_help", {"topic": "control"}),
        "help cockpit": ("jarvis_help", {"topic": "cockpit"}),
        "help status": ("jarvis_help", {"topic": "status"}),
        "help channel health": ("jarvis_help", {"topic": "channel health"}),
        "help channel status": ("jarvis_help", {"topic": "channel status"}),
        "help approval queue": ("jarvis_help", {"topic": "approval queue"}),
        "help pending approvals": ("jarvis_help", {"topic": "pending approvals"}),
        "help risk matrix": ("jarvis_help", {"topic": "risk matrix"}),
        "help morning brief": ("jarvis_help", {"topic": "morning brief"}),
        "help daily brief": ("jarvis_help", {"topic": "daily brief"}),
        "help live proof": ("jarvis_help", {"topic": "live proof"}),
        "help me with calendar": ("capability_map", {"focus": "calendar"}),
        "help me with approvals": ("capability_map", {"focus": "approvals"}),
        "help me choose between red and blue": ("choose_option", {"options": ["red", "blue"]}),
        "morning brief": ("daily_briefing", {}),
    }
    for phrase, expected_action in expected.items():
        plan = planner.plan(phrase)
        actions = [(action.tool_name, action.args) for action in plan.actions]
        if actions != [expected_action]:
            raise SystemExit(f"Help alias misplanned {phrase!r}: {actions!r}")


def main() -> None:
    assert_help_aliases_route_to_help_or_capabilities()
    with TemporaryDirectory(prefix="jarvis-help-") as temp:
        runtime = make_temp_runtime(Path(temp))
        for case, expected_parts in CASES.items():
            result = runtime.handle(case)
            status = "ok" if result.verified else "blocked"
            print(f"[{status}] {case}")
            print(result.response[:1200])
            print()
            if not result.verified:
                raise SystemExit(f"Expected '{case}' to run.")
            for expected in expected_parts:
                if expected not in result.response:
                    raise SystemExit(f"Help output missing expected text for '{case}': {expected}")
            if "approval packet reviewed and verification receipt linked" in result.response:
                raise SystemExit(f"Help output still advertises old approval evidence wording for '{case}'.")
            metadata = result.tool_results[0].metadata
            for key in READ_ONLY_FLAGS:
                if metadata.get(key) is not False:
                    raise SystemExit(f"Help tool unsafe metadata {key}: {metadata}")
            handoff = metadata.get("help_handoff")
            if metadata.get("help_handoff_ready") is not True or not isinstance(handoff, dict):
                raise SystemExit(f"Help tool should expose help_handoff: {metadata}")
            expected_topic = case.split(maxsplit=1)[1] if case.startswith("help ") else "index"
            if case == "jarvis help automation":
                expected_topic = "automation"
            if case in {
                "help cockpit",
                "help status",
                "help contrl",
                "help cokpit",
                "help statuz",
                "help channel health",
                "help channel status",
                "help risk matrix",
                "help live proof",
            }:
                expected_topic = "control"
            if case in {"help approval queue", "help pending approvals"}:
                expected_topic = "approvals"
            if case in {"help morning brief", "help daily brief"}:
                expected_topic = "automation"
            if handoff.get("source") != "jarvis_help" or handoff.get("topic") != expected_topic or metadata.get("topic") != expected_topic:
                raise SystemExit(f"Help handoff topic/source mismatch for {case}: {handoff}")
            if metadata.get("help_status") != "ok" or handoff.get("ready_for_operator") is not True or handoff.get("changed") != []:
                raise SystemExit(f"Help handoff status/changed mismatch for {case}: {handoff}")
            for key in ("authorizes_execution", "authorizes_completion_claim", "approval_granted"):
                if handoff.get(key) is not False:
                    raise SystemExit(f"Help handoff should keep {key}=False for {case}: {handoff}")
            if handoff.get("content_in_handoff") is not False:
                raise SystemExit(f"Help handoff should not carry full help content for {case}: {handoff}")
            if handoff.get("sections") != metadata.get("sections") or handoff.get("entries") != metadata.get("entries"):
                raise SystemExit(f"Help handoff metadata parity failed for {case}: {handoff} / {metadata}")
            if case == "help":
                if handoff.get("index") is not True or handoff.get("sections") < len(CASES) // 2:
                    raise SystemExit(f"Help index handoff should expose topic rows: {handoff}")
                if not handoff.get("section_rows") or not all("command" in row for row in handoff["section_rows"]):
                    raise SystemExit(f"Help index handoff missed section rows: {handoff}")
            else:
                if handoff.get("index") is not False or handoff.get("sections") != 1:
                    raise SystemExit(f"Help topic handoff should expose one topic row for {case}: {handoff}")
            commands = handoff.get("next_commands")
            if not isinstance(commands, list) or "capability map" not in commands or "safety status" not in commands:
                raise SystemExit(f"Help handoff missed command-first next commands for {case}: {handoff}")
            boundaries = handoff.get("boundaries")
            if not isinstance(boundaries, dict) or boundaries.get("read_only") is not True:
                raise SystemExit(f"Help handoff missed read-only boundaries for {case}: {handoff}")
            for key in READ_ONLY_FLAGS:
                if boundaries.get(key) is not False:
                    raise SystemExit(f"Help handoff unsafe boundary {key} for {case}: {handoff}")
            if any(fragment in str(handoff) for fragment in ["/\x55sers/", "/private/", "/var/folders/", "/tmp/"]):
                raise SystemExit(f"Help handoff leaked local path for {case}: {handoff}")


if __name__ == "__main__":
    main()
