from __future__ import annotations

from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory

from jarvis_v2.agent.planner import RuleBasedPlanner
from jarvis_v2.scripts.test_runtime import handle_runtime_case, make_temp_runtime


def test_planner_routes_start_of_day_commands() -> None:
    planner = RuleBasedPlanner()
    daily_plan_cases = [
        "daily plan please",
        "plan my day please",
        "show daily plan",
        "show latest daily plan",
        "today plan please",
        "today's plan please",
        "what is my plan today",
        "what's my plan today",
        "save daily plan please",
        "write daily plan please",
        "save today plan please",
        "write today plan please",
        "save my daily plan",
        "save plan for today",
        # Real gap found live 2026-07-09: "show my daily plan" fell through
        # to chat. Root cause: `_strip_trailing_politeness` strips leading
        # wrapper words including bare "show" (see LEADING_POLITE_WRAPPER_RE),
        # so this phrase reduces to "my daily plan" before the exact-match
        # comparison -- the fix adds that post-strip form to the target set,
        # not the literal "show my daily plan" string, which could never match.
        "show my daily plan",
    ]
    for case in daily_plan_cases:
        date_before = datetime.now().strftime("%Y-%m-%d")
        actions = planner.plan(case).actions
        date_after = datetime.now().strftime("%Y-%m-%d")
        if (
            [action.tool_name for action in actions] != ["daily_plan"]
            or actions[0].args.get("target_date") not in {date_before, date_after}
            or set(actions[0].args) != {"target_date"}
        ):
            raise SystemExit(f"planner missed daily_plan route for {case!r}: {actions!r}")

    morning_startup_cases = [
        "morning startup please",
        "startup brief please",
        "start my day please",
        "show morning startup",
        "show latest morning startup",
        "morning plan please",
        "today startup please",
    ]
    for case in morning_startup_cases:
        actions = planner.plan(case).actions
        if [action.tool_name for action in actions] != ["morning_startup"]:
            raise SystemExit(f"planner missed morning_startup route for {case!r}: {actions!r}")

    save_morning_startup_cases = [
        "save morning startup please",
        "write morning startup please",
        "save startup brief please",
        "write startup brief please",
        "save start my day please",
        "save morning plan please",
    ]
    for case in save_morning_startup_cases:
        actions = planner.plan(case).actions
        if [action.tool_name for action in actions] != ["save_morning_startup"]:
            raise SystemExit(f"planner missed save_morning_startup route for {case!r}: {actions!r}")

    actions = planner.plan("what should i do today please").actions
    if [action.tool_name for action in actions] != ["next_action_packet"]:
        raise SystemExit(f"safe next-action route was stolen: {actions!r}")


def test_daily_plan_enabled_values_fail_closed() -> None:
    with TemporaryDirectory(prefix="jarvis-daily-plan-enabled-fail-closed-") as temp:
        runtime = make_temp_runtime(Path(temp))
        scheduled = runtime.handle("schedule assistant basics")
        if not scheduled.verified:
            raise SystemExit(f"daily-plan enabled fixture could not schedule assistant basics: {scheduled.response}")
        with runtime.store.connect() as conn:
            conn.execute(
                "UPDATE scheduled_jobs SET enabled = 'true' WHERE name IN (?, ?)",
                ("State Snapshot", "Conversation Compaction"),
            )
        result = runtime.handle("daily plan")
        if not result.verified:
            raise SystemExit(f"daily plan should still render with malformed enabled values: {result.response}")
        expected = [
            "State Snapshot [state_snapshot, off]",
            "Conversation Compaction [conversation_compaction, off]",
        ]
        missing = [item for item in expected if item not in result.response]
        if missing:
            raise SystemExit(f"daily plan should fail closed on malformed enabled values: {missing}\n{result.response}")
        forbidden = [
            "State Snapshot [state_snapshot, on]",
            "Conversation Compaction [conversation_compaction, on]",
        ]
        leaked = [item for item in forbidden if item in result.response]
        if leaked:
            raise SystemExit(f"daily plan displayed malformed enabled values as on: {leaked}\n{result.response}")


def main() -> None:
    test_planner_routes_start_of_day_commands()
    test_daily_plan_enabled_values_fail_closed()
    with TemporaryDirectory(prefix="jarvis-daily-plan-") as temp:
        runtime = make_temp_runtime(Path(temp))
        daily_note = Path(temp) / "Vault" / "Jarvis" / "Daily" / f"{datetime.now().strftime('%Y-%m-%d')}.md"
        cases = [
            ("add task review open Jarvis tasks priority high", False),
            ("create goal Build daily planning because stay oriented", False),
            ("add step to goal 1: write a tactical plan builder", False),
            ("schedule assistant basics", False),
            ("run command python3 --version", False),
            ("morning startup", False),
            ("save morning startup", False),
            ("daily plan", False),
        ]
        for case, approved in cases:
            result = handle_runtime_case(runtime, case, approved=approved)
            status = "ok" if result.verified else "blocked"
            print(f"[{status}] {case}")
            print(result.response[:1800])
            print()
            if case == "morning startup":
                required = [
                    "Jarvis morning startup",
                    "Safety first",
                    "Blocked or waiting",
                    "Approval #",
                    "Today queue",
                    "review open Jarvis tasks",
                    "Goal momentum",
                    "Build daily planning",
                    "Suggested first safe move",
                    "approval review",
                ]
                missing = [item for item in required if item not in result.response]
                if missing:
                    raise SystemExit(f"Morning startup missing expected context: {missing}")
            if case == "save morning startup":
                if not daily_note.exists():
                    raise SystemExit("Daily note was not written by save morning startup.")
                note_text = daily_note.read_text(encoding="utf-8")
                required = [
                    "Jarvis Morning Startup",
                    "Jarvis morning startup",
                    "Safety first",
                    "Blocked or waiting",
                    "Approval #",
                    "review open Jarvis tasks",
                ]
                missing = [item for item in required if item not in note_text]
                if missing:
                    raise SystemExit(f"Saved morning startup note missing expected context: {missing}")
            if case == "daily plan":
                required = ["Focus Queue", "Pending Approvals", "Today/Next Automations", "Suggested Order"]
                missing = [item for item in required if item not in result.response]
                if missing:
                    raise SystemExit(f"Daily plan missing expected context: {missing}")


if __name__ == "__main__":
    main()
